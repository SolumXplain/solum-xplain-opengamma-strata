package com.opengamma.strata.pricer.credit;

import java.time.LocalDate;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
import com.opengamma.strata.collect.array.DoubleMatrix;
import com.opengamma.strata.collect.tuple.Pair;
import com.opengamma.strata.data.MarketData;
import com.opengamma.strata.market.curve.CurveInfoType;
import com.opengamma.strata.market.curve.CurveName;
import com.opengamma.strata.market.curve.CurveParameterSize;
import com.opengamma.strata.market.curve.JacobianCalibrationMatrix;
import com.opengamma.strata.market.curve.NodalCurve;
import com.opengamma.strata.market.curve.node.CdsIsdaCreditCurveNode;
import com.opengamma.strata.market.param.CurrencyParameterSensitivities;
import com.opengamma.strata.market.param.ParameterMetadata;
import com.opengamma.strata.market.param.ResolvedTradeParameterMetadata;
import com.opengamma.strata.market.sensitivity.PointSensitivities;
import com.opengamma.strata.math.impl.matrix.CommonsMatrixAlgebra;
import com.opengamma.strata.math.impl.matrix.MatrixAlgebra;
import com.opengamma.strata.pricer.common.PriceType;
import com.opengamma.strata.product.credit.CdsCalibrationTrade;
import com.opengamma.strata.product.credit.CdsQuote;
import com.opengamma.strata.product.credit.ResolvedCdsTrade;
import com.opengamma.strata.product.credit.type.CdsQuoteConvention;

public class XplainCreditCurveCalibrator extends IsdaCompliantCreditCurveCalibrator {

  private static final ArbitrageHandling DEFAULT_ARBITRAGE_HANDLING = ArbitrageHandling.IGNORE;
  private static final AccrualOnDefaultFormula DEFAULT_FORMULA = AccrualOnDefaultFormula.ORIGINAL_ISDA;
  private static final MatrixAlgebra MATRIX_ALGEBRA = new CommonsMatrixAlgebra();
  private static final double ONE_BP = 1.0e-4;

  private final IsdaCdsTradePricer tradePricer;
  private final ArbitrageHandling arbHandling;
  private final AccrualOnDefaultFormula formula;

  protected XplainCreditCurveCalibrator() {
    this(DEFAULT_FORMULA, DEFAULT_ARBITRAGE_HANDLING);
  }

  public XplainCreditCurveCalibrator(AccrualOnDefaultFormula formula) {
    this(formula, DEFAULT_ARBITRAGE_HANDLING);
  }

  public XplainCreditCurveCalibrator(AccrualOnDefaultFormula formula, ArbitrageHandling arbHandling) {
    this.arbHandling = ArgChecker.notNull(arbHandling, "arbHandling");
    this.formula = ArgChecker.notNull(formula, "formula");
    this.tradePricer = new IsdaCdsTradePricer(formula);
  }

  /**
   * Method to calibrate a given credit curve, as well as computing 2 jacobians: (1) ∂ZHRates/∂CreditRates (2) ∂ZHRates/∂Rates
   * NB: the second Jacobian is the purpose for overriding the OG calibrate() method, and is outlined in xplJacobianCalculation():
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
      // nodalCurve = ogJacobianCalculation(legalEntityId, currency, valuationDate, nodalCurve, ratesProvider, quoteConvention, name, nNodes, refData, diag, trades);
      nodalCurve = xplJacobianCalculation(legalEntityId, currency, valuationDate, nodalCurve, ratesProvider, quoteConvention, name, nNodes, refData, diag, trades);
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
    return new FastCreditCurveCalibrator(formula, arbHandling).calibrate(
        calibrationCDSs,
        flactionalSpreads,
        pointsUpfront,
        name,
        valuationDate,
        discountFactors,
        recoveryRates,
        refData);
  }

  // original OG jacobian calculation (use this for testing)
  private NodalCurve ogJacobianCalculation(StandardId legalEntityId,
      Currency currency,
      LocalDate valuationDate,
      NodalCurve nodalCurve,
      ImmutableCreditRatesProvider ratesProvider,
      CdsQuoteConvention quoteConvention,
      CurveName name,
      int nNodes,
      ReferenceData refData,
      double[][] diag,
      ImmutableList<ResolvedCdsTrade> trades) {
    JacobianCalibrationMatrix jacobian = hazardRateByCreditRateCalibrationMatrix(legalEntityId,
        currency,
        valuationDate,
        nodalCurve,
        ratesProvider,
        quoteConvention,
        name,
        nNodes,
        refData,
        diag,
        trades);
    return nodalCurve.withMetadata(nodalCurve.getMetadata().withInfo(CurveInfoType.JACOBIAN, jacobian));
  }

  /**
   * Jacobians generated: (1) ∂ZHRates/∂CreditRates (2) ∂ZHRates/∂Rates
   * @param legalEntityId
   * @param currency
   * @param valuationDate
   * @param nodalCurve
   * @param ratesProvider
   * @param quoteConvention
   * @param name
   * @param nNodes
   * @param refData
   * @param diag
   * @param trades
   * @return
   */
  private NodalCurve xplJacobianCalculation(StandardId legalEntityId,
      Currency currency,
      LocalDate valuationDate,
      NodalCurve nodalCurve,
      ImmutableCreditRatesProvider ratesProvider,
      CdsQuoteConvention quoteConvention,
      CurveName name,
      int nNodes,
      ReferenceData refData,
      double[][] diag,
      ImmutableList<ResolvedCdsTrade> trades) {

    // (1)
    JacobianCalibrationMatrix dZhrDCrJacobian = hazardRateByCreditRateCalibrationMatrix(legalEntityId, currency, valuationDate, nodalCurve, ratesProvider, quoteConvention, name, nNodes, refData, diag, trades);
    nodalCurve = nodalCurve.withMetadata(nodalCurve.getMetadata().withInfo(CurveInfoType.JACOBIAN, dZhrDCrJacobian));

    // (2)
    JacobianCalibrationMatrix dZhrDCrJacobian2 = hazardRateBySwapRateCalibrationMatrix(legalEntityId, currency, valuationDate, nodalCurve, ratesProvider, quoteConvention, name, nNodes, refData, diag, trades);
    nodalCurve = nodalCurve.withMetadata(nodalCurve.getMetadata().withInfo(CurveInfoType.JACOBIAN_2, dZhrDCrJacobian2));

    return nodalCurve;
  }

  // ∂ZHRates/∂CreditRates
  private JacobianCalibrationMatrix hazardRateByCreditRateCalibrationMatrix(StandardId legalEntityId,
      Currency currency,
      LocalDate valuationDate,
      NodalCurve nodalCurve,
      ImmutableCreditRatesProvider ratesProvider,
      CdsQuoteConvention quoteConvention,
      CurveName name,
      int nNodes,
      ReferenceData refData,
      double[][] diag,
      ImmutableList<ResolvedCdsTrade> trades) {
    LegalEntitySurvivalProbabilities creditCurve = LegalEntitySurvivalProbabilities.of(
        legalEntityId, IsdaCreditDiscountFactors.of(currency, valuationDate, nodalCurve));
    ImmutableCreditRatesProvider ratesProviderNew = ratesProvider.toBuilder()
        .creditCurves(ImmutableMap.of(Pair.of(legalEntityId, currency), creditCurve))
        .build();
    Function<ResolvedCdsTrade, DoubleArray> sensiFunc = quoteConvention.equals(CdsQuoteConvention.PAR_SPREAD) ?
        getParSpreadSensitivityFunction(ratesProviderNew, name, currency, refData) :
        getPointsUpfrontSensitivityFunction(ratesProviderNew, name, currency, refData);
    DoubleMatrix sensi = DoubleMatrix.ofArrayObjects(nNodes, nNodes, i -> sensiFunc.apply(trades.get(i)));
    sensi = (DoubleMatrix) MATRIX_ALGEBRA.multiply(DoubleMatrix.ofUnsafe(diag), sensi);
    return JacobianCalibrationMatrix.of(
        ImmutableList.of(CurveParameterSize.of(name, nNodes)), MATRIX_ALGEBRA.getInverse(sensi));
  }

  /**
   *   ∂ZHRates/∂Rates = ∂ZHRates/∂ZCRates * ∂ZCRates/∂Rates
   *      -> ∂ZHRates/∂ZCRates = - ∂F/ZCRates * [∂F/∂ZHRates]^-1     (implicit function theorem)
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
   * @return ∂ZHRates/∂Rates
   */
  private JacobianCalibrationMatrix hazardRateBySwapRateCalibrationMatrix(StandardId legalEntityId,
      Currency currency,
      LocalDate valuationDate,
      NodalCurve nodalCurve,
      ImmutableCreditRatesProvider ratesProvider,
      CdsQuoteConvention quoteConvention,
      CurveName name,
      int nNodes,
      ReferenceData refData,
      double[][] diag,
      ImmutableList<ResolvedCdsTrade> trades) {

    // OG
    LegalEntitySurvivalProbabilities creditCurve = LegalEntitySurvivalProbabilities.of(
        legalEntityId, IsdaCreditDiscountFactors.of(currency, valuationDate, nodalCurve));

    // OG
    ImmutableCreditRatesProvider ratesProviderNew = ratesProvider.toBuilder()
        .creditCurves(ImmutableMap.of(Pair.of(legalEntityId, currency), creditCurve))
        .build();

    // extract discount curve
    // TODO: add safety check
    IsdaCreditDiscountFactors discountFactors = ((IsdaCreditDiscountFactors) ratesProvider.discountFactors(currency));
    NodalCurve discountCurve = ((IsdaCreditDiscountFactors) ratesProvider.discountFactors(currency)).getCurve();

    // extract discount curve ZC rates
    DoubleArray discountCurveZcRates = discountCurve.getYValues();


    // generate M1
    DoubleMatrix m1Matrix = generateM1Matrix(trades, currency, ratesProviderNew, discountCurveZcRates, refData);

    // generate M2


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    Function<ResolvedCdsTrade, DoubleArray> sensiFunc = quoteConvention.equals(CdsQuoteConvention.PAR_SPREAD) ?
        getParSpreadSensitivityFunction(ratesProviderNew, name, currency, refData) :
        getPointsUpfrontSensitivityFunction(ratesProviderNew, name, currency, refData);
    DoubleMatrix sensi = DoubleMatrix.ofArrayObjects(nNodes, nNodes, i -> sensiFunc.apply(trades.get(i)));
    sensi = (DoubleMatrix) MATRIX_ALGEBRA.multiply(DoubleMatrix.ofUnsafe(diag), sensi);
    return JacobianCalibrationMatrix.of(
        ImmutableList.of(CurveParameterSize.of(name, nNodes)), MATRIX_ALGEBRA.getInverse(sensi));
  }

  // ∂F/dZCRates
  // -> when we revalue each trade by shifting ZC value, the element in the matrix will show the change in pv
  // -> i.e., PV(shifted) - PV(unshifted) => PV(shifted) - 0 => PV(shifted)
  private DoubleMatrix generateM1Matrix(ImmutableList<ResolvedCdsTrade> trades, Currency currency, ImmutableCreditRatesProvider creditRatesProvider, DoubleArray discountCurveZcRates, ReferenceData referenceData) {
    int nDiscountCurveNodes = discountCurveZcRates.size();

    Function<ResolvedCdsTrade, DoubleArray> nodeZcTradePvSensitivityFunction =
        resolvedCdsTrade -> shiftedPvDeltasForTrade(creditRatesProvider, resolvedCdsTrade, currency, referenceData);

    return DoubleMatrix.ofArrayObjects(trades.size(), nDiscountCurveNodes, i -> nodeZcTradePvSensitivityFunction.apply(trades.get(i)));
  }

  /**
   *
   * @return array where each element represents trade's PV sensitivity to a ZC rate in curve
   */
    private DoubleArray shiftedPvDeltasForTrade(ImmutableCreditRatesProvider creditRatesProvider, ResolvedCdsTrade cdsTrade, Currency currency, ReferenceData referenceData) {
      IsdaCreditDiscountFactors creditDiscountFactors = (IsdaCreditDiscountFactors) creditRatesProvider.discountFactors(currency);
      NodalCurve discountCurve = creditDiscountFactors.getCurve();
      DoubleArray discountCurveZcRates = discountCurve.getYValues();
      return DoubleArray.of(discountCurveZcRates.size(), (int index) ->
          getZcShiftPriceDelta(cdsTrade, currency, creditRatesProvider, shiftedDiscountCurveZcRates(discountCurveZcRates, index), referenceData));
    }

  /**
   *
   * @return PV delta for a trade when a given ZC value is shifted
   */
  // TODO: review CLEAN price
  // TODO: it seems price before shift is not 0, so for now, need to look at price before - price after shifting a ZC value
  private double getZcShiftPriceDelta(ResolvedCdsTrade trade, Currency currency, ImmutableCreditRatesProvider creditRatesProvider, DoubleArray shiftedDiscountCurveZcRates, ReferenceData referenceData) {
    double unshiftedPrice = tradePricer.price(trade, creditRatesProvider, PriceType.CLEAN, referenceData);
    ImmutableCreditRatesProvider ratesProviderWithShiftedZcValue = ratesProviderWithShiftedZcValue(currency, creditRatesProvider, shiftedDiscountCurveZcRates);
    double shiftedPrice = tradePricer.price(trade, ratesProviderWithShiftedZcValue, PriceType.CLEAN, referenceData);
    return shiftedPrice - unshiftedPrice;
  }

  /**
   *
   * @param discountCurveZcRates unshifted ZC rates from a curve
   * @param index the index of the rate we want to shift
   * @return input ZC rates where element at index is shifted by 1 basis point
   */
  private DoubleArray shiftedDiscountCurveZcRates(DoubleArray discountCurveZcRates, int index) {
    return discountCurveZcRates.with(index, discountCurveZcRates.get(index) + ONE_BP);
  }

  /**
   *
   * @param currency calibration currency
   * @param creditRatesProvider original rates provider, holds curves and their respective x and y values
   * @param shiftedCurveZcRates array of zc rates, where one has been shifted
   * @return rates provider with overridden y values for discount curve(s)
   */
  private ImmutableCreditRatesProvider ratesProviderWithShiftedZcValue(Currency currency, ImmutableCreditRatesProvider creditRatesProvider, DoubleArray shiftedCurveZcRates) {
    IsdaCreditDiscountFactors creditDiscountFactors = (IsdaCreditDiscountFactors) creditRatesProvider.discountFactors(currency);
    NodalCurve discountCurve = creditDiscountFactors.getCurve();

    // override discount curve y values
    NodalCurve shiftedDiscountCurve = discountCurve.withYValues(shiftedCurveZcRates);

    // override discount curve in rates provider
    // TODO: what if more than one currency in existing map? needs to be a cleaner way of doing this!
    IsdaCreditDiscountFactors newCreditDiscountFactors = IsdaCreditDiscountFactors.of(currency, creditDiscountFactors.getValuationDate(), shiftedDiscountCurve);
    ImmutableMap<Currency, CreditDiscountFactors> newDiscountCurvesMap = ImmutableMap.of(currency, newCreditDiscountFactors);
    return creditRatesProvider.toBuilder().discountCurves(newDiscountCurvesMap).build();
  }





  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

  private Function<ResolvedCdsTrade, DoubleArray> getParSpreadSensitivityFunction(
      CreditRatesProvider ratesProvider,
      CurveName curveName,
      Currency currency,
      ReferenceData refData) {

    Function<ResolvedCdsTrade, DoubleArray> func = new Function<ResolvedCdsTrade, DoubleArray>() {
      @Override
      public DoubleArray apply(ResolvedCdsTrade trade) {
        PointSensitivities point = tradePricer.parSpreadSensitivity(trade, ratesProvider, refData);
        return ratesProvider.parameterSensitivity(point).getSensitivity(curveName, currency).getSensitivity();
      }
    };
    return func;
  }

  private Function<ResolvedCdsTrade, DoubleArray> getPointsUpfrontSensitivityFunction(
      CreditRatesProvider ratesProvider,
      CurveName curveName,
      Currency currency,
      ReferenceData refData) {

    Function<ResolvedCdsTrade, DoubleArray> func = new Function<ResolvedCdsTrade, DoubleArray>() {
      @Override
      public DoubleArray apply(ResolvedCdsTrade trade) {
        PointSensitivities point = tradePricer.priceSensitivity(trade, ratesProvider, refData);
        return ratesProvider.parameterSensitivity(point).getSensitivity(curveName, currency).getSensitivity();
      }
    };
    return func;
  }
}
