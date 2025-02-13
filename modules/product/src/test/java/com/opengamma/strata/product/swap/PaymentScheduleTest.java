/*
 * Copyright (C) 2014 - present by OpenGamma Inc. and the OpenGamma group of companies
 *
 * Please see distribution for license.
 */
package com.opengamma.strata.product.swap;

import static com.opengamma.strata.basics.date.BusinessDayConventions.MODIFIED_FOLLOWING;
import static com.opengamma.strata.basics.date.HolidayCalendarIds.GBLO;
import static com.opengamma.strata.basics.date.HolidayCalendarIds.SAT_SUN;
import static com.opengamma.strata.basics.schedule.Frequency.P1M;
import static com.opengamma.strata.basics.schedule.Frequency.P2M;
import static com.opengamma.strata.basics.schedule.Frequency.P3M;
import static com.opengamma.strata.basics.schedule.Frequency.TERM;
import static com.opengamma.strata.basics.schedule.RollConventions.DAY_5;
import static com.opengamma.strata.basics.schedule.RollConventions.EOM;
import static com.opengamma.strata.collect.TestHelper.assertSerialization;
import static com.opengamma.strata.collect.TestHelper.coverBeanEquals;
import static com.opengamma.strata.collect.TestHelper.coverImmutableBean;
import static com.opengamma.strata.collect.TestHelper.date;
import static com.opengamma.strata.product.swap.CompoundingMethod.NONE;
import static com.opengamma.strata.product.swap.CompoundingMethod.STRAIGHT;
import static com.opengamma.strata.product.swap.PaymentRelativeTo.PERIOD_END;
import static com.opengamma.strata.product.swap.PaymentRelativeTo.PERIOD_START;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatExceptionOfType;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import java.time.LocalDate;
import java.util.Optional;

import org.junit.jupiter.api.Test;

import com.opengamma.strata.basics.ReferenceData;
import com.opengamma.strata.basics.currency.Currency;
import com.opengamma.strata.basics.currency.CurrencyAmount;
import com.opengamma.strata.basics.date.BusinessDayAdjustment;
import com.opengamma.strata.basics.date.BusinessDayConventions;
import com.opengamma.strata.basics.date.DayCounts;
import com.opengamma.strata.basics.date.DaysAdjustment;
import com.opengamma.strata.basics.date.HolidayCalendarIds;
import com.opengamma.strata.basics.index.IborIndices;
import com.opengamma.strata.basics.schedule.Frequency;
import com.opengamma.strata.basics.schedule.PeriodicSchedule;
import com.opengamma.strata.basics.schedule.RollConventions;
import com.opengamma.strata.basics.schedule.Schedule;
import com.opengamma.strata.basics.schedule.ScheduleException;
import com.opengamma.strata.basics.schedule.SchedulePeriod;
import com.opengamma.strata.basics.schedule.StubConvention;
import com.opengamma.strata.product.TradeInfo;
import com.opengamma.strata.product.common.PayReceive;

/**
 * Test.
 */
public class PaymentScheduleTest {

  private static final ReferenceData REF_DATA = ReferenceData.standard();
  private static final LocalDate DATE_01_05 = date(2014, 1, 5);
  private static final LocalDate DATE_01_06 = date(2014, 1, 6);
  private static final LocalDate DATE_01_08 = date(2014, 1, 8);
  private static final LocalDate DATE_02_05 = date(2014, 2, 5);
  private static final LocalDate DATE_03_05 = date(2014, 3, 5);
  private static final LocalDate DATE_04_04 = date(2014, 4, 4);
  private static final LocalDate DATE_04_05 = date(2014, 4, 5);
  private static final LocalDate DATE_04_07 = date(2014, 4, 7);
  private static final LocalDate DATE_04_30 = date(2014, 4, 30);
  private static final LocalDate DATE_05_05 = date(2014, 5, 5);
  private static final LocalDate DATE_05_06 = date(2014, 5, 6);
  private static final LocalDate DATE_05_30 = date(2014, 5, 30);
  private static final LocalDate DATE_05_31 = date(2014, 5, 31);
  private static final BusinessDayAdjustment BDA = BusinessDayAdjustment.of(MODIFIED_FOLLOWING, SAT_SUN);

  private static final SchedulePeriod ACCRUAL1STUB = SchedulePeriod.of(DATE_01_08, DATE_02_05, DATE_01_08, DATE_02_05);
  private static final SchedulePeriod ACCRUAL1 = SchedulePeriod.of(DATE_01_06, DATE_02_05, DATE_01_05, DATE_02_05);
  private static final SchedulePeriod ACCRUAL2 = SchedulePeriod.of(DATE_02_05, DATE_03_05, DATE_02_05, DATE_03_05);
  private static final SchedulePeriod ACCRUAL3 = SchedulePeriod.of(DATE_03_05, DATE_04_07, DATE_03_05, DATE_04_05);
  private static final SchedulePeriod ACCRUAL4 = SchedulePeriod.of(DATE_04_07, DATE_05_06, DATE_04_05, DATE_05_05);
  private static final SchedulePeriod ACCRUAL_5 = SchedulePeriod.of(DATE_04_30, DATE_05_31);
  private static final SchedulePeriod ACCRUAL3STUB = SchedulePeriod.of(DATE_03_05, DATE_04_04, DATE_03_05, DATE_04_04);
  private static final SchedulePeriod ACCRUAL4STUB = SchedulePeriod.of(DATE_04_07, DATE_04_30, DATE_04_05, DATE_04_30);
  private static final Schedule ACCRUAL_SCHEDULE_SINGLE = Schedule.builder()
      .periods(ACCRUAL1)
      .frequency(P1M)
      .rollConvention(RollConventions.DAY_5)
      .build();
  private static final Schedule ACCRUAL_SCHEDULE_TERM = Schedule.builder()
      .periods(SchedulePeriod.of(DATE_01_06, DATE_04_07, DATE_01_05, DATE_04_05))
      .frequency(TERM)
      .rollConvention(RollConventions.NONE)
      .build();
  private static final Schedule ACCRUAL_SCHEDULE = Schedule.builder()
      .periods(ACCRUAL1, ACCRUAL2, ACCRUAL3)
      .frequency(P1M)
      .rollConvention(DAY_5)
      .build();
  private static final Schedule ACCRUAL_SCHEDULE_STUBS = Schedule.builder()
      .periods(ACCRUAL1STUB, ACCRUAL2, ACCRUAL3STUB)
      .frequency(P1M)
      .rollConvention(DAY_5)
      .build();
  private static final Schedule ACCRUAL_SCHEDULE_INITIAL_STUB = Schedule.builder()
      .periods(ACCRUAL1STUB, ACCRUAL2, ACCRUAL3, ACCRUAL4)
      .frequency(P1M)
      .rollConvention(DAY_5)
      .build();
  private static final Schedule ACCRUAL_SCHEDULE_FINAL_STUB = Schedule.builder()
      .periods(ACCRUAL1, ACCRUAL2, ACCRUAL3STUB)
      .frequency(P1M)
      .rollConvention(DAY_5)
      .build();
  private static final Schedule ACCRUAL_SCHEDULE_FINAL_STUB_4PERIODS = Schedule.builder()
      .periods(ACCRUAL1, ACCRUAL2, ACCRUAL3, ACCRUAL4STUB)
      .frequency(P1M)
      .rollConvention(DAY_5)
      .build();
  private static final Schedule ACCRUAL_TERM_HOLIDAY_END = Schedule.builder()
      .periods(ACCRUAL_5)
      .frequency(TERM)
      .rollConvention(EOM)
      .build();

  //-------------------------------------------------------------------------
  @Test
  public void test_builder_ensureDefaults() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P1M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .build();
    assertThat(test.getPaymentFrequency()).isEqualTo(P1M);
    assertThat(test.getBusinessDayAdjustment()).isEqualTo(Optional.empty());
    assertThat(test.getPaymentDateOffset()).isEqualTo(DaysAdjustment.ofBusinessDays(2, GBLO));
    assertThat(test.getPaymentRelativeTo()).isEqualTo(PERIOD_END);
    assertThat(test.getCompoundingMethod()).isEqualTo(NONE);
  }

  //-------------------------------------------------------------------------
  @Test
  public void test_createSchedule_sameFrequency() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P1M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA);
    assertThat(schedule).isEqualTo(ACCRUAL_SCHEDULE);
  }

  @Test
  public void test_createSchedule_singleAccrualPeriod() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P1M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE_SINGLE, REF_DATA);
    assertThat(schedule).isEqualTo(ACCRUAL_SCHEDULE_SINGLE);
  }

  @Test
  public void test_createSchedule_term() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(TERM)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE_TERM, REF_DATA);
    assertThat(schedule).isEqualTo(ACCRUAL_SCHEDULE_TERM);
  }

  @Test
  public void test_createSchedule_term_holiday_end() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(TERM)
        .businessDayAdjustment(BDA)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(0, GBLO))
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_TERM_HOLIDAY_END, REF_DATA);
    assertThat(schedule).isNotEqualTo(ACCRUAL_TERM_HOLIDAY_END);
    assertThat(schedule.getEndDate()).isEqualTo(DATE_05_30);
  }

  @Test
  public void test_createSchedule_term_badFirstRegular() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(TERM)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .firstRegularStartDate(DATE_05_05)
        .build();
    assertThatIllegalArgumentException()
        .isThrownBy(() -> test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA));
  }

  @Test
  public void test_createSchedule_term_badLastRegular() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(TERM)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .lastRegularEndDate(DATE_05_05)
        .build();
    assertThatIllegalArgumentException()
        .isThrownBy(() -> test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA));
  }

  @Test
  public void test_createSchedule_fullMerge() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P3M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(SchedulePeriod.of(DATE_01_06, DATE_04_07, DATE_01_05, DATE_04_05))
        .frequency(P3M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  @Test
  public void test_createSchedule_partMergeForwards() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(
            SchedulePeriod.of(DATE_01_06, DATE_03_05, DATE_01_05, DATE_03_05),
            SchedulePeriod.of(DATE_03_05, DATE_04_07, DATE_03_05, DATE_04_05))
        .frequency(P2M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  @Test
  public void test_createSchedule_initialStubPartMergeBackwards() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE_INITIAL_STUB, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(
            ACCRUAL1STUB,
            SchedulePeriod.of(DATE_02_05, DATE_03_05, DATE_02_05, DATE_03_05),
            SchedulePeriod.of(DATE_03_05, DATE_05_06, DATE_03_05, DATE_05_05))
        .frequency(P2M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  @Test
  public void test_createSchedule_finalStubFullMerge() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE_FINAL_STUB, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(
            SchedulePeriod.of(DATE_01_06, DATE_03_05, DATE_01_05, DATE_03_05),
            ACCRUAL3STUB)
        .frequency(P2M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  @Test
  public void test_createSchedule_dualStub() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE_STUBS, REF_DATA);
    assertThat(schedule).isEqualTo(ACCRUAL_SCHEDULE_STUBS.toBuilder().frequency(P2M).build());
  }

  //-------------------------------------------------------------------------
  @Test
  public void test_createSchedule_firstAndLastDate() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P3M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .firstRegularStartDate(DATE_01_06)
        .lastRegularEndDate(DATE_04_07)
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(SchedulePeriod.of(DATE_01_06, DATE_04_07, DATE_01_05, DATE_04_05))
        .frequency(P3M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  @Test
  public void test_createSchedule_firstAndLastDate_validInitialStub() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .firstRegularStartDate(DATE_02_05)
        .lastRegularEndDate(DATE_04_07)
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(
            SchedulePeriod.of(DATE_01_06, DATE_02_05, DATE_01_05, DATE_02_05),
            SchedulePeriod.of(DATE_02_05, DATE_04_07, DATE_02_05, DATE_04_05))
        .frequency(P2M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  @Test
  public void test_createSchedule_firstAndLastDate_invalidInitialStub() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .firstRegularStartDate(DATE_01_06)
        .lastRegularEndDate(DATE_04_07)
        .build();
    assertThatExceptionOfType(ScheduleException.class)
        .isThrownBy(() -> test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA));
  }

  @Test
  public void test_createSchedule_firstAndLastDate_initialAccrualStub() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .firstRegularStartDate(DATE_03_05)
        .lastRegularEndDate(DATE_05_05)
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE_INITIAL_STUB, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(
            SchedulePeriod.of(DATE_01_08, DATE_03_05, DATE_01_08, DATE_03_05),
            SchedulePeriod.of(DATE_03_05, DATE_05_06, DATE_03_05, DATE_05_05))
        .frequency(P2M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  //-------------------------------------------------------------------------
  @Test
  public void test_createSchedule_firstDate() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P3M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .firstRegularStartDate(DATE_01_06)
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(SchedulePeriod.of(DATE_01_06, DATE_04_07, DATE_01_05, DATE_04_05))
        .frequency(P3M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  @Test
  public void test_createSchedule_firstDate_validInitialStub() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .firstRegularStartDate(DATE_02_05)
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(
            SchedulePeriod.of(DATE_01_06, DATE_02_05, DATE_01_05, DATE_02_05),
            SchedulePeriod.of(DATE_02_05, DATE_04_07, DATE_02_05, DATE_04_05))
        .frequency(P2M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  @Test
  public void test_createSchedule_firstDate_invalidInitialStub() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .firstRegularStartDate(DATE_01_06)
        .build();
    assertThatExceptionOfType(ScheduleException.class)
        .isThrownBy(() -> test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA));
  }

  @Test
  public void test_createSchedule_firstDate_initialAccrualStub() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .firstRegularStartDate(DATE_03_05)
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE_INITIAL_STUB, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(
            SchedulePeriod.of(DATE_01_08, DATE_03_05, DATE_01_08, DATE_03_05),
            SchedulePeriod.of(DATE_03_05, DATE_05_06, DATE_03_05, DATE_05_05))
        .frequency(P2M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  //-------------------------------------------------------------------------
  @Test
  public void test_createSchedule_lastDate() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P3M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .lastRegularEndDate(DATE_04_05)
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(SchedulePeriod.of(DATE_01_06, DATE_04_07, DATE_01_05, DATE_04_05))
        .frequency(P3M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  @Test
  public void test_createSchedule_lastDate_validFinalStub() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .lastRegularEndDate(DATE_03_05)
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(
            SchedulePeriod.of(DATE_01_06, DATE_03_05, DATE_01_05, DATE_03_05),
            SchedulePeriod.of(DATE_03_05, DATE_04_07, DATE_03_05, DATE_04_05))
        .frequency(P2M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  @Test
  public void test_createSchedule_lastDate_invalidFinalStub() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .lastRegularEndDate(DATE_04_05)
        .build();
    assertThatExceptionOfType(ScheduleException.class)
        .isThrownBy(() -> test.createSchedule(ACCRUAL_SCHEDULE, REF_DATA));
  }

  @Test
  public void test_createSchedule_lastDate_finalAccrualStub() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P2M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .lastRegularEndDate(DATE_03_05)
        .build();
    Schedule schedule = test.createSchedule(ACCRUAL_SCHEDULE_FINAL_STUB_4PERIODS, REF_DATA);
    Schedule expected = Schedule.builder()
        .periods(
            SchedulePeriod.of(DATE_01_06, DATE_03_05, DATE_01_05, DATE_03_05),
            SchedulePeriod.of(DATE_03_05, DATE_04_30, DATE_03_05, DATE_04_30))
        .frequency(P2M)
        .rollConvention(DAY_5)
        .build();
    assertThat(schedule).isEqualTo(expected);
  }

  @Test
  public void testSingleDayInitialStub() {
    PeriodicSchedule accrualSchedule = PeriodicSchedule.builder()
        .startDate(LocalDate.of(2021, 8, 25))
        .endDate(LocalDate.of(2022, 8, 26))
        .frequency(Frequency.P6M)
        .businessDayAdjustment(BusinessDayAdjustment.of(
            BusinessDayConventions.MODIFIED_FOLLOWING,
            HolidayCalendarIds.JPTO))
        .startDateBusinessDayAdjustment(BusinessDayAdjustment.NONE)
        .endDateBusinessDayAdjustment(BusinessDayAdjustment.of(
            BusinessDayConventions.MODIFIED_FOLLOWING,
            HolidayCalendarIds.JPTO))
        .stubConvention(StubConvention.SHORT_INITIAL)
        .rollConvention(RollConventions.DAY_26)
        .build();

    PaymentSchedule paymentSchedule = PaymentSchedule.builder()
        .paymentFrequency(Frequency.P6M)
        .businessDayAdjustment(BusinessDayAdjustment.of(
            BusinessDayConventions.MODIFIED_FOLLOWING,
            HolidayCalendarIds.JPTO))
        .paymentRelativeTo(PaymentRelativeTo.PERIOD_END)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(0, HolidayCalendarIds.JPTO))
        .compoundingMethod(CompoundingMethod.NONE)
        .build();

    NotionalSchedule notionalSchedule = NotionalSchedule.of(CurrencyAmount.of(Currency.JPY, 1000000000));

    Swap swap = Swap.builder()
        .legs(
            RateCalculationSwapLeg.builder()
                .payReceive(PayReceive.RECEIVE)
                .accrualSchedule(accrualSchedule)
                .paymentSchedule(paymentSchedule)
                .notionalSchedule(notionalSchedule)
                .calculation(FixedRateCalculation.of(0.001, DayCounts.ACT_365F))
                .build(),
            RateCalculationSwapLeg.builder()
                .payReceive(PayReceive.PAY)
                .accrualSchedule(accrualSchedule)
                .paymentSchedule(paymentSchedule)
                .notionalSchedule(notionalSchedule)
                .calculation(IborRateCalculation.of(IborIndices.JPY_TIBOR_JAPAN_6M))
                .build())
        .build();
    SwapTrade swapTrade = SwapTrade.of(TradeInfo.empty(), swap);
    ResolvedSwapTrade resolvedSwapTrade = swapTrade.resolve(REF_DATA);

    RatePaymentPeriod pp0 = (RatePaymentPeriod) resolvedSwapTrade.getProduct().getLegs().get(0).getPaymentPeriods().get(0);
    RateAccrualPeriod ap0 = pp0.getAccrualPeriods().get(0);
    assertThat(ap0.getStartDate()).isEqualTo(LocalDate.of(2021, 8, 25));
    assertThat(ap0.getUnadjustedStartDate()).isEqualTo(LocalDate.of(2021, 8, 25));
    assertThat(ap0.getEndDate()).isEqualTo(LocalDate.of(2021, 8, 26));
    assertThat(ap0.getUnadjustedEndDate()).isEqualTo(LocalDate.of(2021, 8, 26));

    RatePaymentPeriod pp1 = (RatePaymentPeriod) resolvedSwapTrade.getProduct().getLegs().get(0).getPaymentPeriods().get(1);
    RateAccrualPeriod ap1 = pp1.getAccrualPeriods().get(0);
    assertThat(ap1.getStartDate()).isEqualTo(LocalDate.of(2021, 8, 26));
    assertThat(ap1.getUnadjustedStartDate()).isEqualTo(LocalDate.of(2021, 8, 26));
    assertThat(ap1.getEndDate()).isEqualTo(LocalDate.of(2022, 2, 28));
    assertThat(ap1.getUnadjustedEndDate()).isEqualTo(LocalDate.of(2022, 2, 26));

    RatePaymentPeriod pp2 = (RatePaymentPeriod) resolvedSwapTrade.getProduct().getLegs().get(0).getPaymentPeriods().get(2);
    RateAccrualPeriod ap2 = pp2.getAccrualPeriods().get(0);
    assertThat(ap2.getStartDate()).isEqualTo(LocalDate.of(2022, 2, 28));
    assertThat(ap2.getUnadjustedStartDate()).isEqualTo(LocalDate.of(2022, 2, 26));
    assertThat(ap2.getEndDate()).isEqualTo(LocalDate.of(2022, 8, 26));
    assertThat(ap2.getUnadjustedEndDate()).isEqualTo(LocalDate.of(2022, 8, 26));
  }

  //-------------------------------------------------------------------------
  @Test
  public void coverage() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P1M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .build();
    coverImmutableBean(test);
    PaymentSchedule test2 = PaymentSchedule.builder()
        .paymentFrequency(P3M)
        .businessDayAdjustment(BDA)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(3, GBLO))
        .paymentRelativeTo(PERIOD_START)
        .compoundingMethod(STRAIGHT)
        .firstRegularStartDate(DATE_01_06)
        .lastRegularEndDate(DATE_02_05)
        .build();
    coverBeanEquals(test, test2);
  }

  @Test
  public void test_serialization() {
    PaymentSchedule test = PaymentSchedule.builder()
        .paymentFrequency(P3M)
        .paymentDateOffset(DaysAdjustment.ofBusinessDays(2, GBLO))
        .build();
    assertSerialization(test);
  }

}
