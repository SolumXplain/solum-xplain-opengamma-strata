/**
 * Copyright (C) 2014 - present by OpenGamma Inc. and the OpenGamma group of companies
 *
 * Please see distribution for license.
 */
package com.opengamma.platform.source;

import static com.opengamma.platform.source.SearchMatchStatus.FULL;
import static com.opengamma.platform.source.SearchMatchStatus.PARTIAL;

import java.io.Serializable;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

import org.joda.beans.Bean;
import org.joda.beans.BeanBuilder;
import org.joda.beans.BeanDefinition;
import org.joda.beans.ImmutableBean;
import org.joda.beans.JodaBeanUtils;
import org.joda.beans.MetaProperty;
import org.joda.beans.Property;
import org.joda.beans.PropertyDefinition;
import org.joda.beans.impl.direct.DirectFieldsBeanBuilder;
import org.joda.beans.impl.direct.DirectMetaBean;
import org.joda.beans.impl.direct.DirectMetaProperty;
import org.joda.beans.impl.direct.DirectMetaPropertyMap;

import com.google.common.collect.ImmutableSet;
import com.opengamma.collect.id.StandardId;

/**
 * The result of a search of a {@code SearchableSourceProvider}
 * <p>
 * Results can either be an exact match for the search criteria
 * provided or a partial match. The latter is intended for cases
 * where implementers do not have access to all the data needed to do
 * the filtering in their underlying data store. By providing the
 * partial results a separate filter operation can take place later
 * to refine the results to the correct set.
 */
@BeanDefinition(builderScope = "private")
public final class SearchResult
    implements ImmutableBean, Serializable {

  /**
   * The collection of identifiers that match or partially match the
   * original search request.
   */
  @PropertyDefinition(validate = "notNull")
  private final ImmutableSet<StandardId> matchingIds;
  /**
   * The match status indicating whether all the search criteria
   * have been satisfied in producing the results.
   */
  @PropertyDefinition(validate = "notNull")
  private final SearchMatchStatus matchStatus;

  //-------------------------------------------------------------------------
  /**
   * Creates a SearchResult with a collection of matching identifiers and
   * an indication that the results do not satisfy all the specified
   * search criteria.
   *
   * @param matchingIds  the collections of identifiers that have been
   *   determined to partially match the search criteria
   * @return the new SearchResult
   */
  public static SearchResult partialMatch(Iterable<StandardId> matchingIds) {
    return new SearchResult(ImmutableSet.copyOf(matchingIds), PARTIAL);
  }

  /**
   * Creates a SearchResult with a collection of matching identifiers and
   * an indication that the results satisfy all the specified
   * search criteria.
   *
   * @param matchingIds  the collections of identifiers that have been
   *   determined to fully match the search criteria
   * @return the new SearchResult
   */
  public static SearchResult fullMatch(Iterable<StandardId> matchingIds) {
    return new SearchResult(ImmutableSet.copyOf(matchingIds), FULL);
  }

  //------------------------- AUTOGENERATED START -------------------------
  ///CLOVER:OFF
  /**
   * The meta-bean for {@code SearchResult}.
   * @return the meta-bean, not null
   */
  public static SearchResult.Meta meta() {
    return SearchResult.Meta.INSTANCE;
  }

  static {
    JodaBeanUtils.registerMetaBean(SearchResult.Meta.INSTANCE);
  }

  /**
   * The serialization version id.
   */
  private static final long serialVersionUID = 1L;

  private SearchResult(
      Set<StandardId> matchingIds,
      SearchMatchStatus matchStatus) {
    JodaBeanUtils.notNull(matchingIds, "matchingIds");
    JodaBeanUtils.notNull(matchStatus, "matchStatus");
    this.matchingIds = ImmutableSet.copyOf(matchingIds);
    this.matchStatus = matchStatus;
  }

  @Override
  public SearchResult.Meta metaBean() {
    return SearchResult.Meta.INSTANCE;
  }

  @Override
  public <R> Property<R> property(String propertyName) {
    return metaBean().<R>metaProperty(propertyName).createProperty(this);
  }

  @Override
  public Set<String> propertyNames() {
    return metaBean().metaPropertyMap().keySet();
  }

  //-----------------------------------------------------------------------
  /**
   * Gets the collection of identifiers that match or partially match the
   * original search request.
   * @return the value of the property, not null
   */
  public ImmutableSet<StandardId> getMatchingIds() {
    return matchingIds;
  }

  //-----------------------------------------------------------------------
  /**
   * Gets the match status indicating whether all the search criteria
   * have been satisfied in producing the results.
   * @return the value of the property, not null
   */
  public SearchMatchStatus getMatchStatus() {
    return matchStatus;
  }

  //-----------------------------------------------------------------------
  @Override
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }
    if (obj != null && obj.getClass() == this.getClass()) {
      SearchResult other = (SearchResult) obj;
      return JodaBeanUtils.equal(getMatchingIds(), other.getMatchingIds()) &&
          JodaBeanUtils.equal(getMatchStatus(), other.getMatchStatus());
    }
    return false;
  }

  @Override
  public int hashCode() {
    int hash = getClass().hashCode();
    hash = hash * 31 + JodaBeanUtils.hashCode(getMatchingIds());
    hash = hash * 31 + JodaBeanUtils.hashCode(getMatchStatus());
    return hash;
  }

  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder(96);
    buf.append("SearchResult{");
    buf.append("matchingIds").append('=').append(getMatchingIds()).append(',').append(' ');
    buf.append("matchStatus").append('=').append(JodaBeanUtils.toString(getMatchStatus()));
    buf.append('}');
    return buf.toString();
  }

  //-----------------------------------------------------------------------
  /**
   * The meta-bean for {@code SearchResult}.
   */
  public static final class Meta extends DirectMetaBean {
    /**
     * The singleton instance of the meta-bean.
     */
    static final Meta INSTANCE = new Meta();

    /**
     * The meta-property for the {@code matchingIds} property.
     */
    @SuppressWarnings({"unchecked", "rawtypes" })
    private final MetaProperty<ImmutableSet<StandardId>> matchingIds = DirectMetaProperty.ofImmutable(
        this, "matchingIds", SearchResult.class, (Class) ImmutableSet.class);
    /**
     * The meta-property for the {@code matchStatus} property.
     */
    private final MetaProperty<SearchMatchStatus> matchStatus = DirectMetaProperty.ofImmutable(
        this, "matchStatus", SearchResult.class, SearchMatchStatus.class);
    /**
     * The meta-properties.
     */
    private final Map<String, MetaProperty<?>> metaPropertyMap$ = new DirectMetaPropertyMap(
        this, null,
        "matchingIds",
        "matchStatus");

    /**
     * Restricted constructor.
     */
    private Meta() {
    }

    @Override
    protected MetaProperty<?> metaPropertyGet(String propertyName) {
      switch (propertyName.hashCode()) {
        case -2026007173:  // matchingIds
          return matchingIds;
        case 1644523031:  // matchStatus
          return matchStatus;
      }
      return super.metaPropertyGet(propertyName);
    }

    @Override
    public BeanBuilder<? extends SearchResult> builder() {
      return new SearchResult.Builder();
    }

    @Override
    public Class<? extends SearchResult> beanType() {
      return SearchResult.class;
    }

    @Override
    public Map<String, MetaProperty<?>> metaPropertyMap() {
      return metaPropertyMap$;
    }

    //-----------------------------------------------------------------------
    /**
     * The meta-property for the {@code matchingIds} property.
     * @return the meta-property, not null
     */
    public MetaProperty<ImmutableSet<StandardId>> matchingIds() {
      return matchingIds;
    }

    /**
     * The meta-property for the {@code matchStatus} property.
     * @return the meta-property, not null
     */
    public MetaProperty<SearchMatchStatus> matchStatus() {
      return matchStatus;
    }

    //-----------------------------------------------------------------------
    @Override
    protected Object propertyGet(Bean bean, String propertyName, boolean quiet) {
      switch (propertyName.hashCode()) {
        case -2026007173:  // matchingIds
          return ((SearchResult) bean).getMatchingIds();
        case 1644523031:  // matchStatus
          return ((SearchResult) bean).getMatchStatus();
      }
      return super.propertyGet(bean, propertyName, quiet);
    }

    @Override
    protected void propertySet(Bean bean, String propertyName, Object newValue, boolean quiet) {
      metaProperty(propertyName);
      if (quiet) {
        return;
      }
      throw new UnsupportedOperationException("Property cannot be written: " + propertyName);
    }

  }

  //-----------------------------------------------------------------------
  /**
   * The bean-builder for {@code SearchResult}.
   */
  private static final class Builder extends DirectFieldsBeanBuilder<SearchResult> {

    private Set<StandardId> matchingIds = ImmutableSet.of();
    private SearchMatchStatus matchStatus;

    /**
     * Restricted constructor.
     */
    private Builder() {
    }

    //-----------------------------------------------------------------------
    @Override
    public Object get(String propertyName) {
      switch (propertyName.hashCode()) {
        case -2026007173:  // matchingIds
          return matchingIds;
        case 1644523031:  // matchStatus
          return matchStatus;
        default:
          throw new NoSuchElementException("Unknown property: " + propertyName);
      }
    }

    @SuppressWarnings("unchecked")
    @Override
    public Builder set(String propertyName, Object newValue) {
      switch (propertyName.hashCode()) {
        case -2026007173:  // matchingIds
          this.matchingIds = (Set<StandardId>) newValue;
          break;
        case 1644523031:  // matchStatus
          this.matchStatus = (SearchMatchStatus) newValue;
          break;
        default:
          throw new NoSuchElementException("Unknown property: " + propertyName);
      }
      return this;
    }

    @Override
    public Builder set(MetaProperty<?> property, Object value) {
      super.set(property, value);
      return this;
    }

    @Override
    public Builder setString(String propertyName, String value) {
      setString(meta().metaProperty(propertyName), value);
      return this;
    }

    @Override
    public Builder setString(MetaProperty<?> property, String value) {
      super.setString(property, value);
      return this;
    }

    @Override
    public Builder setAll(Map<String, ? extends Object> propertyValueMap) {
      super.setAll(propertyValueMap);
      return this;
    }

    @Override
    public SearchResult build() {
      return new SearchResult(
          matchingIds,
          matchStatus);
    }

    //-----------------------------------------------------------------------
    @Override
    public String toString() {
      StringBuilder buf = new StringBuilder(96);
      buf.append("SearchResult.Builder{");
      buf.append("matchingIds").append('=').append(JodaBeanUtils.toString(matchingIds)).append(',').append(' ');
      buf.append("matchStatus").append('=').append(JodaBeanUtils.toString(matchStatus));
      buf.append('}');
      return buf.toString();
    }

  }

  ///CLOVER:ON
  //-------------------------- AUTOGENERATED END --------------------------
}
