package com.opengamma.strata.pricer.credit;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.ImmutableMap;
import com.opengamma.strata.basics.ReferenceData;
import com.opengamma.strata.basics.StandardId;
import com.opengamma.strata.basics.currency.Currency;
import com.opengamma.strata.basics.date.DayCount;
import com.opengamma.strata.collect.ArgChecker;
import com.opengamma.strata.collect.Guavate;
import com.opengamma.strata.collect.array.DoubleArray;
import com.opengamma.strata.collect.tuple.Pair;
import com.opengamma.strata.data.MarketData;
import com.opengamma.strata.market.curve.CurveName;
import com.opengamma.strata.market.curve.IsdaCreditCurveDefinition;
import com.opengamma.strata.market.curve.NodalCurve;
import com.opengamma.strata.market.curve.node.CdsIsdaCreditCurveNode;
import com.opengamma.strata.market.param.CurrencyParameterSensitivities;
import com.opengamma.strata.market.param.ParameterMetadata;
import com.opengamma.strata.market.param.ResolvedTradeParameterMetadata;
import com.opengamma.strata.pricer.common.PriceType;
import com.opengamma.strata.product.credit.CdsCalibrationTrade;
import com.opengamma.strata.product.credit.CdsQuote;
import com.opengamma.strata.product.credit.ResolvedCdsTrade;

import com.opengamma.strata.product.credit.type.CdsQuoteConvention;

import java.time.LocalDate;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class XplainIsdaCompliantCreditCurveCalibrator extends IsdaCompliantCreditCurveCalibrator {

  private final IsdaCdsTradePricer tradePricer;

  public XplainIsdaCompliantCreditCurveCalibrator(AccrualOnDefaultFormula formula,
      IsdaCdsTradePricer tradePricer) {
    super(formula);
    this.tradePricer = tradePricer;
  }

  public XplainIsdaCompliantCreditCurveCalibrator(AccrualOnDefaultFormula formula,
      ArbitrageHandling arbHandling, IsdaCdsTradePricer tradePricer) {
    super(formula, arbHandling);
    this.tradePricer = tradePricer;
  }

  /**
   * Method to calibrate a given credit curve, as well as computing 2 jacobians: (1) ∂ZHRates/∂CreditRates (2) ∂ZHRates/∂Rates
   * NB:
   *   the second Jacobian is the purpose for overriding the OG calibrate() method, and is outlined below:
   *      -> ∂ZHRates/∂Rates = ∂ZHRates/∂ZCRates * ∂ZCRates/∂Rates
   *      -> ∂ZHRates/∂ZCRates = - ∂F/ZCRates * [∂F/∂ZHRates]^-1     (implict function theorem)
   *         where F(ZCRates,ZHRates) is a vector-valued function that returns the value of CDS trades
   *   the process by which ∂ZHRates/∂ZCRates is constructed is as follows:
   *      -> initialise m-n matrix (m = number of trades, n = number of nodes in discount curve)
   *          -> for each ZCRate, bump and revalue each trade, and produce matrix => M1
   *      -> initialise m-n matrix (m = number of trades, n = number of nodes in credit curve)
   *          -> for each ZHRate, bump and revalue each trade, and produce matrix
   *          -> invert matrix => M2
   *      -> ∂ZHRates/∂ZCRates = -M1 * M2
   *   the jacobian matrix of ∂ZHRates/∂Rates is then generated via ∂ZHRates/∂ZCRates * ∂ZCRates/∂ZRates
   *      where ∂ZCRates/∂ZRates is extracted from the previously calibrated discount curve
   *
   * @param curveNodes list of a single credit curve's nodes
   * @param name curve's name
   * @param marketData raw market data extraction
   * @param ratesProvider recovery rates, discount factors, will be populated with calibrated credit curves
   * @param definitionDayCount curve definition day count
   * @param definitionCurrency curve definition currency
   * @param computeJacobian if true, compute the jacobian
   * @param storeTrade if true, store all generated curve node-trades in returned curve metadata
   * @param refData holiday calendars etc
   * @return LegalEntitySurvivalProbabilities - holds calibrated curve and jacobian in survivalProbabilities
   */
  @Override
  public LegalEntitySurvivalProbabilities calibrate(
      List<CdsIsdaCreditCurveNode> curveNodes,
      CurveName name,
      MarketData marketData,
      ImmutableCreditRatesProvider ratesProvider,
      DayCount definitionDayCount,
      Currency definitionCurrency,
      boolean computeJacobian,
      boolean storeTrade,
      ReferenceData refData) {
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // OG Start //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Iterator<StandardId> legalEntities =
        curveNodes.stream().map(CdsIsdaCreditCurveNode::getLegalEntityId).collect(Collectors.toSet()).iterator();
    StandardId legalEntityId = legalEntities.next();
    ArgChecker.isFalse(legalEntities.hasNext(), "legal entity must be common to curve nodes");
    Iterator<Currency> currencies =
        curveNodes.stream().map(n -> n.getTemplate().getConvention().getCurrency()).collect(Collectors.toSet()).iterator();
    Currency currency = currencies.next();
    ArgChecker.isFalse(currencies.hasNext(), "currency must be common to curve nodes");
    ArgChecker.isTrue(definitionCurrency.equals(currency),
        "curve definition currency must be the same as the currency of CDS");
    Iterator<CdsQuoteConvention> quoteConventions =
        curveNodes.stream().map(n -> n.getQuoteConvention()).collect(Collectors.toSet()).iterator();
    CdsQuoteConvention quoteConvention = quoteConventions.next();
    ArgChecker.isFalse(quoteConventions.hasNext(), "quote convention must be common to curve nodes");
    LocalDate valuationDate = marketData.getValuationDate();
    ArgChecker.isTrue(valuationDate.equals(marketData.getValuationDate()),
        "ratesProvider and marketDate must be based on the same valuation date");
    CreditDiscountFactors discountFactors = ratesProvider.discountFactors(currency);
    ArgChecker.isTrue(definitionDayCount.equals(discountFactors.getDayCount()),
        "credit curve and discount curve must be based on the same day count convention");
    RecoveryRates recoveryRates = ratesProvider.recoveryRates(legalEntityId);

    int nNodes = curveNodes.size();
    double[] coupons = new double[nNodes];
    double[] pufs = new double[nNodes];
    double[][] diag = new double[nNodes][nNodes];
    Builder<ResolvedCdsTrade> tradesBuilder = ImmutableList.builder();
    for (int i = 0; i < nNodes; i++) {
      CdsCalibrationTrade tradeCalibration = curveNodes.get(i).trade(1d, marketData, refData);
      ResolvedCdsTrade trade = tradeCalibration.getUnderlyingTrade().resolve(refData);
      tradesBuilder.add(trade);
      double[] temp = getStandardQuoteForm(
          trade,
          tradeCalibration.getQuote(),
          valuationDate,
          discountFactors,
          recoveryRates,
          computeJacobian,
          refData);
      coupons[i] = temp[0];
      pufs[i] = temp[1];
      diag[i][i] = temp[2];
    }
    ImmutableList<ResolvedCdsTrade> trades = tradesBuilder.build();
    NodalCurve nodalCurve = calibrate(
        trades,
        DoubleArray.ofUnsafe(coupons),
        DoubleArray.ofUnsafe(pufs),
        name,
        valuationDate,
        discountFactors,
        recoveryRates,
        refData);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // OG End ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (computeJacobian) {
      // implementation here
      System.out.println("Computing jacobians...");
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // OG Start //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ImmutableList<ParameterMetadata> parameterMetadata;
    if (storeTrade) {
      parameterMetadata = IntStream.range(0, nNodes)
          .mapToObj(n -> ResolvedTradeParameterMetadata.of(trades.get(n), curveNodes.get(n).getLabel()))
          .collect(Guavate.toImmutableList());
    } else {
      parameterMetadata = IntStream.range(0, nNodes)
          .mapToObj(n -> curveNodes.get(n).metadata(trades.get(n).getProduct().getProtectionEndDate()))
          .collect(Guavate.toImmutableList());
    }
    nodalCurve = nodalCurve.withMetadata(nodalCurve.getMetadata().withParameterMetadata(parameterMetadata));

    return LegalEntitySurvivalProbabilities.of(
        legalEntityId, IsdaCreditDiscountFactors.of(currency, valuationDate, nodalCurve));
  }

  @Override
  public NodalCurve calibrate(List<ResolvedCdsTrade> calibrationCDSs, DoubleArray flactionalSpreads,
      DoubleArray pointsUpfront, CurveName name, LocalDate valuationDate,
      CreditDiscountFactors discountFactors, RecoveryRates recoveryRates, ReferenceData refData) {
    // TODO: review calibrator options
    return FastCreditCurveCalibrator.standard().calibrate(
        calibrationCDSs,
        flactionalSpreads,
        pointsUpfront,
        name,
        valuationDate,
        discountFactors,
        recoveryRates,
        refData);
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // OG ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  private double[] getStandardQuoteForm(ResolvedCdsTrade calibrationCds, CdsQuote marketQuote, LocalDate valuationDate,
      CreditDiscountFactors discountFactors, RecoveryRates recoveryRates, boolean computeJacobian, ReferenceData refData) {

    double[] res = new double[3];
    res[2] = 1d;
    if (marketQuote.getQuoteConvention().equals(CdsQuoteConvention.PAR_SPREAD)) {
      res[0] = marketQuote.getQuotedValue();
    } else if (marketQuote.getQuoteConvention().equals(CdsQuoteConvention.QUOTED_SPREAD)) {
      double qSpread = marketQuote.getQuotedValue();
      CurveName curveName = CurveName.of("quoteConvertCurve");
      NodalCurve tempCreditCurve = calibrate(
          ImmutableList.of(calibrationCds),
          DoubleArray.of(qSpread),
          DoubleArray.of(0d),
          curveName,
          valuationDate,
          discountFactors,
          recoveryRates,
          refData);
      Currency currency = calibrationCds.getProduct().getCurrency();
      StandardId legalEntityId = calibrationCds.getProduct().getLegalEntityId();
      ImmutableCreditRatesProvider rates = ImmutableCreditRatesProvider.builder()
          .valuationDate(valuationDate)
          .discountCurves(ImmutableMap.of(currency, discountFactors))
          .recoveryRateCurves(ImmutableMap.of(legalEntityId, recoveryRates))
          .creditCurves(
              ImmutableMap.of(
                  Pair.of(legalEntityId, currency),
                  LegalEntitySurvivalProbabilities.of(
                      legalEntityId,
                      IsdaCreditDiscountFactors.of(currency, valuationDate, tempCreditCurve))))
          .build();
      res[0] = calibrationCds.getProduct().getFixedRate();
      res[1] = tradePricer.price(calibrationCds, rates, PriceType.CLEAN, refData);
      if (computeJacobian) {
        CurrencyParameterSensitivities pufSensi =
            rates.parameterSensitivity(tradePricer.priceSensitivity(calibrationCds, rates, refData));
        CurrencyParameterSensitivities spSensi =
            rates.parameterSensitivity(tradePricer.parSpreadSensitivity(calibrationCds, rates, refData));
        res[2] = spSensi.getSensitivity(curveName, currency).getSensitivity().get(0) /
            pufSensi.getSensitivity(curveName, currency).getSensitivity().get(0);
      }
    } else if (marketQuote.getQuoteConvention().equals(CdsQuoteConvention.POINTS_UPFRONT)) {
      res[0] = calibrationCds.getProduct().getFixedRate();
      res[1] = marketQuote.getQuotedValue();
    } else {
      throw new IllegalArgumentException("Unknown CDSQuoteConvention type " + marketQuote.getClass());
    }
    return res;
  }


}
