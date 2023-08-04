package com.opengamma.strata.pricer.credit;

import com.opengamma.strata.basics.ReferenceData;
import com.opengamma.strata.data.MarketData;
import com.opengamma.strata.market.curve.IsdaCreditCurveDefinition;

public class XplainIsdaCompliantCreditCurveCalibrator {

  // override of com/opengamma/strata/pricer/credit/IsdaCompliantCreditCurveCalibrator.java:151

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
   * @param curveDefinition calibration parameters (e.g., computeJacobian flag, list of nodes)
   * @param marketData raw market data extraction
   * @param ratesProvider recovery rates, discount factors, will be populated with calibrated credit curves
   * @param refData holiday calendars etc
   * @return currency, valuationDate, calibrated curve with
   */
  public LegalEntitySurvivalProbabilities calibrate(
      IsdaCreditCurveDefinition curveDefinition,
      MarketData marketData,
      ImmutableCreditRatesProvider ratesProvider,
      ReferenceData refData) {
    return null;
  }

}
