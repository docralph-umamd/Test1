import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =====================================
# PAGE CONFIGURATION
# =====================================
st.set_page_config(
    page_title="Healthcare Analytics Hub",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM CSS FOR BRANDING
# =====================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# DATA LOADING & CACHING
# =====================================
@st.cache_data(ttl=3600)
def load_and_prepare_data(file):
    """Load and prepare data with caching"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        df.columns = df.columns.str.strip().str.lower()
        
        # Ensure numeric approved column
        if "approved" in df.columns:
            df["approved"] = pd.to_numeric(df["approved"], errors="coerce").fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def calculate_outliers(df, multiplier=1.5):
    """Calculate outliers using IQR method"""
    Q1 = df['approved'].quantile(0.25)
    Q3 = df['approved'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    core_data = df[(df['approved'] >= lower_bound) & (df['approved'] <= upper_bound)]
    outliers = df[(df['approved'] < lower_bound) | (df['approved'] > upper_bound)]
    
    return core_data, outliers, upper_bound

@st.cache_data
def detect_anomalies(df):
    """Simple anomaly detection using statistical methods"""
    mean = df['approved'].mean()
    std = df['approved'].std()
    threshold = mean + (3 * std)
    
    df['is_anomaly'] = df['approved'] > threshold
    return df

# =====================================
# INITIALIZE SESSION STATE
# =====================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'selected_service_type' not in st.session_state:
    st.session_state.selected_service_type = None
if 'iqr_multiplier' not in st.session_state:
    st.session_state.iqr_multiplier = 1.5

# =====================================
# SIDEBAR - GLOBAL CONTROLS
# =====================================
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=HealthCare+Analytics", use_container_width=True)
    st.title("🎛️ Control Panel")
    
    # File Upload
    st.header("📁 Data Source")
    uploaded_file = st.file_uploader(
        "Upload Claims Data",
        type=["xlsx", "csv"],
        help="Upload Excel or CSV file with claims data"
    )
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            df = load_and_prepare_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(df):,} records")
    
    # Navigation
    st.header("📊 Navigation")
    
    if st.session_state.data_loaded:
        page = st.radio(
            "Select Analysis",
            [
                "🏠 Executive Dashboard",
                "🧩 Service Mix Analysis",
                "🧍‍♂️ Top Utilizers",
                "📈 Trend Analysis",
                "🎯 Predictive Insights",
                "🔍 Data Explorer"
            ],
            label_visibility="collapsed"
        )
        
        # Global Filters
        st.divider()
        st.subheader("🔧 Global Filters")
        
        df = st.session_state.df
        
        # Amount range filter
        if 'approved' in df.columns:
            min_amt, max_amt = int(df['approved'].min()), int(df['approved'].max())
            amount_range = st.slider(
                "Approved Amount Range (₱)",
                min_value=min_amt,
                max_value=max_amt,
                value=(min_amt, max_amt),
                format="₱%d"
            )
        
        # Service type filter
        loatype_cols = [c for c in df.columns if 'loatype' in c.lower()]
        if loatype_cols:
            loatype_col = loatype_cols[0]
            service_types = ['All'] + sorted(df[loatype_col].dropna().unique().tolist())
            selected_service = st.multiselect(
                "Service Types",
                options=service_types[1:],
                default=None
            )
        
        # Month filter if available
        month_cols = [c for c in df.columns if c.lower() in ['month', 'monthname', 'admissionmonth']]
        if month_cols:
            month_col = month_cols[0]
            months = ['All'] + sorted(df[month_col].dropna().unique().tolist())
            selected_months = st.multiselect(
                "Months",
                options=months[1:],
                default=None
            )
        
        # Apply filters
        filtered_df = df.copy()
        if 'approved' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['approved'] >= amount_range[0]) & 
                (filtered_df['approved'] <= amount_range[1])
            ]
        if loatype_cols and selected_service:
            filtered_df = filtered_df[filtered_df[loatype_col].isin(selected_service)]
        if month_cols and selected_months:
            filtered_df = filtered_df[filtered_df[month_col].isin(selected_months)]
        
        st.session_state.filtered_df = filtered_df
        
        # Filter summary
        st.info(f"📊 {len(filtered_df):,} records after filtering")
        
        # Outlier sensitivity
        st.divider()
        st.subheader("⚙️ Advanced Settings")
        st.session_state.iqr_multiplier = st.slider(
            "Outlier Sensitivity",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Higher = fewer outliers detected (IQR multiplier)"
        )
    else:
        st.info("👆 Upload a file to begin analysis")
        page = None

# =====================================
# MAIN CONTENT AREA
# =====================================
if not st.session_state.data_loaded:
    st.markdown('<p class="main-header">🏥 Healthcare Analytics Hub</p>', unsafe_allow_html=True)
    st.subheader("Utilization Management Intelligence Platform")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **Executive Dashboard**\nReal-time KPIs and cost metrics")
    with col2:
        st.info("🧩 **Service Mix Analysis**\nUnderstand utilization patterns")
    with col3:
        st.info("🎯 **Predictive Insights**\nForecast trends and anomalies")
    
    st.divider()
    st.markdown("""
    ### Getting Started
    1. **Upload your claims data** using the sidebar (Excel or CSV format)
    2. **Navigate** through different analysis modules
    3. **Apply filters** to drill down into specific segments
    4. **Export insights** for reporting and decision-making
    
    #### Expected Data Columns:
    - `approved` (required): Approved claim amount
    - `loatype`: Service type (e.g., Inpatient, Outpatient)
    - `month` or `monthname`: Transaction month
    - `policyno` or `cardnotext`: Member identifier
    - Additional columns: `providername`, `gender`, `general_tiering`, etc.
    """)
    
else:
    df = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.df
    
    # =====================================
    # PAGE 1: EXECUTIVE DASHBOARD
    # =====================================
    if page == "🏠 Executive Dashboard":
        st.markdown('<p class="main-header">🏠 Executive Dashboard</p>', unsafe_allow_html=True)
        st.caption(f"📅 Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        
        # Determine the current fiscal year (assuming fiscal year starts in January)
        current_year = datetime.now().year
        fiscal_year = current_year if datetime.now().month >= 1 else current_year - 1

        # Filter DataFrame for fiscal year 2025
        df_2025 = df[df['year'] == 2025]  # Assuming there's a 'year' column in the DataFrame
        df_sply = df[df['year'] == fiscal_year - 1]  # Data for the previous fiscal year

        # Top KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        total_approved = df_2025['approved'].sum()
        avg_claim = df_2025['approved'].mean()
        total_claims = df_2025['claimno'].nunique()  
        total_availers = df_2025['policyno'].nunique() if 'policyno' in df.columns else None

        # Calculate SPLY
        total_approved_sply = df_sply['approved'].sum() if 'approved' in df_sply.columns else 0
        avg_claim_sply = df_sply['approved'].mean() if 'approved' in df_sply.columns else 0
        total_claims_sply = df_sply['claimno'].nunique() if 'claimno' in df_sply.columns else 0
        total_availers_sply = df_sply['policyno'].nunique() if 'policyno' in df_sply.columns else 0

        # Calculate percentage change for approved
        approved_change = (
            ((total_approved - total_approved_sply) / total_approved_sply * 100)
            if total_approved_sply > 0 else 0
        )
        
        # Calculate percentage change for other metrics
        claims_change = (
            ((total_claims - total_claims_sply) / total_claims_sply * 100)
            if total_claims_sply > 0 else 0
        )
        
        availers_change = (
            ((total_availers - total_availers_sply) / total_availers_sply * 100)
            if total_availers_sply > 0 else 0
        )
        
        avg_claim_change = (
            ((avg_claim - avg_claim_sply) / avg_claim_sply * 100)
            if avg_claim_sply > 0 else 0
        )
        # Calculate period over period if month column exists
        month_cols = [c for c in df.columns if c.lower() in ['month', 'monthname', 'admissionmonth']]
        if month_cols:
            month_col = month_cols[0]
            latest_month = df[month_col].iloc[-1] if len(df) > 0 else None
            if latest_month:
                current = df[df[month_col] == latest_month]['approved'].sum()
                previous_months = df[df[month_col] != latest_month][month_col].unique()
                if len(previous_months) > 0:
                    previous_month = previous_months[-1]
                    previous = df[df[month_col] == previous_month]['approved'].sum()
                    mom_change = ((current - previous) / previous * 100) if previous > 0 else 0
                else:
                    mom_change = 0
            else:
                mom_change = 0
        else:
            mom_change = 0
        
        with col1:
            st.metric(
                "💰 Total Approved",
                f"₱{total_approved:,.0f}",
                delta=f"{approved_change:+.1f}% vs. SPLY"
        )

        with col2:
            st.metric(
                "📊 Total Claims",
                f"{total_claims:,}",
                delta=f"{claims_change:+.1f}% vs. SPLY"
            )

        with col3:
            st.metric(
                "👥 Total Availers",
                f"{total_availers:,}",  # Display unique availer count
                delta=f"{availers_change:+.1f}% vs. SPLY"
            )

        with col4:
            st.metric(
                "📈 Average Claim",
                f"₱{avg_claim:,.0f}",
                delta=f"{avg_claim_change:+.1f}% vs. SPLY"
            )

        
        st.divider()
        
        # Get top 200 members by total approved amount
        # Find claim identifier column
        claim_id_cols = [c for c in df_2025.columns if 'claim' in c.lower() and ('no' in c.lower() or 'id' in c.lower())]
        
        if claim_id_cols:
            claim_col = claim_id_cols[0]
            top200_agg = df_2025.groupby('policyno').agg({
                'approved': 'sum',
                claim_col: 'nunique'
            }).reset_index()
            top200_agg.columns = ['policyno', 'approved', 'claim_count']
        else:
            # If no claim column found, just count rows as claims
            top200_agg = df_2025.groupby('policyno').agg({
                'approved': ['sum', 'count']
            }).reset_index()
            top200_agg.columns = ['policyno', 'approved', 'claim_count']
        
        top200 = top200_agg.nlargest(200, 'approved').copy()
        
        core_data, outliers, upper_bound = calculate_outliers(
            top200, 
            st.session_state.iqr_multiplier
        )
        
        # Distribution visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Claims Distribution (Top 200)")
            
            # Create interactive histogram with Plotly
            fig = px.histogram(
                top200,
                x='approved',
                nbins=50,
                title="Distribution of Top 200 Claims",
                labels={'approved': 'Approved Amount (₱)', 'count': 'Frequency'},
                color_discrete_sequence=['#1f77b4']
            )

            # Add mean and median lines with annotations
            mean_val = core_data['approved'].mean()
            median_val = core_data['approved'].median()

            # Add mean line with background color
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: ₱{mean_val:,.0f}",
                annotation_position="top right",
                annotation=dict(
                    bgcolor="rgba(255, 0, 0, 0.3)",  # Light red background
                    bordercolor="red",
                    borderwidth=1,
                    font=dict(color="white")  # White text color
                )
            )

            # Add median line with background color
            fig.add_vline(
                x=median_val,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Median: ₱{median_val:,.0f}",
                annotation_position="top left",
                annotation=dict(
                    bgcolor="rgba(0, 128, 0, 0.3)",  # Light green background
                    bordercolor="green",
                    borderwidth=1,
                    font=dict(color="white")  # White text color
                )
            )

            # Adjust the layout for better visibility
            fig.update_layout(
                showlegend=False,
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📈 Key Statistics")
            
            st.metric("Mean", f"₱{core_data['approved'].mean():,.0f}")
            st.metric("Median", f"₱{core_data['approved'].median():,.0f}")
            st.metric("Std Dev", f"₱{core_data['approved'].std():,.0f}")
            st.metric("Max", f"₱{core_data['approved'].max():,.0f}")
            st.metric("Min", f"₱{core_data['approved'].min():,.0f}")
            
            # Outlier info
            st.divider()
            st.error(f"⚠️ **{len(outliers)} Outliers Detected**")
            st.caption(f"Threshold: ₱{upper_bound:,.0f}")

        st.divider()
        st.subheader("📋 Top 200 Members - Detailed Breakdown")

        # Get the list of top 200 policy numbers
        top200_policies = top200['policyno'].tolist()

        # Filter original data for these members only
        df_top200_detail = df_2025[df_2025['policyno'].isin(top200_policies)].copy()

        # Check if loatype column exists
        loatype_cols = [c for c in df_top200_detail.columns if 'loatype' in c.lower()]

        if loatype_cols:
            loatype_col = loatype_cols[0]
            
            # Create detailed summary with LOA type breakdown
            detailed_summary = df_top200_detail.groupby('policyno').agg({
                'approved': ['sum', 'mean', 'max'],
                claim_col: 'nunique',  # Using the claim_col identified earlier
                loatype_col: lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'  # Most frequent service type
            }).reset_index()
            
            detailed_summary.columns = [
                'Member ID',
                'Total Approved',
                'Avg Claim Amount',
                'Highest Single Claim',
                'Total Claims',
                'Primary Service Type'
            ]
            
            # Sort by total approved and reset index
            detailed_summary = detailed_summary.sort_values('Total Approved', ascending=False)
            detailed_summary.index = range(1, len(detailed_summary) + 1)
            
            # Display
            st.dataframe(
                detailed_summary.style.format({
                    'Total Approved': '₱{:,.0f}',
                    'Avg Claim Amount': '₱{:,.0f}',
                    'Highest Single Claim': '₱{:,.0f}',
                    'Total Claims': '{:,.0f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Add LOA type distribution for top 200
            st.subheader("🧩 Service Type Distribution - Top 200 Members")
            
            loatype_dist = df_top200_detail.groupby(loatype_col).agg({
                'approved': 'sum',
                claim_col: 'nunique'
            }).reset_index()
            loatype_dist.columns = ['Service Type', 'Total Approved', 'Claim Count']
            loatype_dist = loatype_dist.sort_values('Total Approved', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    loatype_dist,
                    x='Total Approved',
                    y='Service Type',
                    orientation='h',
                    title='Top 200 Members: Spend by Service Type',
                    color='Total Approved',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    loatype_dist,
                    values='Claim Count',
                    names='Service Type',
                    title='Top 200 Members: Claims by Service Type',
                    hole=0.3
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display the breakdown table
            st.dataframe(
                loatype_dist.style.format({
                    'Total Approved': '₱{:,.0f}',
                    'Claim Count': '{:,.0f}'
                }),
                use_container_width=True
            )

        else:
            # If no LOA type column, show basic summary
            st.info("💡 LOA Type column not found. Showing basic summary only.")

        st.divider()

        # Prepare comprehensive report
        if loatype_cols:
            # Get all claims for top 200 members with full details
            export_cols = ['policyno', 'approved', claim_col]
            
            # Add optional columns if they exist
            for optional in [loatype_col, 'gender', 'general_tiering', 'providername', 'admissiondate']:
                if optional in df_top200_detail.columns:
                    export_cols.append(optional)
            
            export_df = df_top200_detail[export_cols].copy()
            export_df = export_df.sort_values(['policyno', 'approved'], ascending=[True, False])
            
            st.download_button(
                "📥 Download Top 200 Members - Full Claims Detail",
                export_df.to_csv(index=False).encode('utf-8'),
                "top_200_members_full_detail.csv",
                "text/csv",
                help="Download all claim-level details for the top 200 members"
            )
        
        # Service Mix Overview
        st.divider()
        st.subheader("🧩 Service Mix Overview")
        
        loatype_cols = [c for c in df.columns if 'loatype' in c.lower()]
        if loatype_cols:
            loatype_col = loatype_cols[0]
            
            service_agg = df.groupby(loatype_col)['approved'].agg(['sum', 'count']).reset_index()
            service_agg.columns = ['Service Type', 'Total Approved', 'Claim Count']
            service_agg = service_agg.sort_values('Total Approved', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Treemap
                fig = px.treemap(
                    service_agg,
                    path=['Service Type'],
                    values='Total Approved',
                    title='Service Type by Total Spend',
                    color='Total Approved',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart
                fig = px.pie(
                    service_agg,
                    values='Claim Count',
                    names='Service Type',
                    title='Claim Volume by Service Type',
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Executive Insights
        st.divider()
        st.subheader("💡 Executive Insights")
        
        volatility = (core_data['approved'].std() / core_data['approved'].mean()) * 100
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>🎯 Cost Structure Analysis</h4>
        <ul>
            <li><strong>Average Claim:</strong> ₱{mean_val:,.0f} | <strong>Median:</strong> ₱{median_val:,.0f}</li>
            <li><strong>Volatility Index:</strong> {volatility:.1f}% — {'High variability suggests uneven case severity' if volatility > 50 else 'Stable cost structure'}</li>
            <li><strong>High-Cost Cases:</strong> {len(outliers)} claims exceeded ₱{upper_bound:,.0f}, requiring targeted review</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if loatype_cols:
            top_service = service_agg.iloc[0]
            st.markdown(f"""
            <div class="insight-box">
            <h4>📊 Service Type Concentration</h4>
            <ul>
                <li><strong>Dominant Category:</strong> {top_service['Service Type']} accounts for ₱{top_service['Total Approved']:,.0f} ({top_service['Total Approved']/total_approved*100:.1f}% of total spend)</li>
                <li><strong>Claim Volume:</strong> {top_service['Claim Count']:,} claims ({top_service['Claim Count']/total_claims*100:.1f}% of total)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # =====================================
    # PAGE 2: SERVICE MIX ANALYSIS
    # =====================================
    elif page == "🧩 Service Mix Analysis":
        st.markdown('<p class="main-header">🧩 Service Mix Analysis</p>', unsafe_allow_html=True)
        
        loatype_cols = [c for c in df.columns if 'loatype' in c.lower()]
        if not loatype_cols:
            st.error("❌ No 'LOAtype' column found in the data")
            st.stop()
        
        loatype_col = loatype_cols[0]
        
        # Month-over-Month Comparison
        month_cols = [c for c in df.columns if c.lower() in ['month', 'monthname', 'admissionmonth']]
        
        if month_cols:
            month_col = month_cols[0]
            available_months = sorted(df[month_col].dropna().unique().tolist())
            
            st.subheader("📆 Month-over-Month Comparison")
            
            col1, col2 = st.columns(2)
            with col1:
                month_a = st.selectbox("Select Month A", available_months, index=max(0, len(available_months)-2))
            with col2:
                month_b = st.selectbox("Select Month B", available_months, index=len(available_months)-1)
            
            # Data for both months
            df_a = df[df[month_col] == month_a]
            df_b = df[df[month_col] == month_b]
            
            agg_a = df_a.groupby(loatype_col)['approved'].sum().reset_index()
            agg_b = df_b.groupby(loatype_col)['approved'].sum().reset_index()
            
            total_a = agg_a['approved'].sum()
            total_b = agg_b['approved'].sum()
            pct_change = ((total_b - total_a) / total_a * 100) if total_a > 0 else 0
            
            # Top services
            top_a = agg_a.sort_values('approved', ascending=False).iloc[0] if len(agg_a) > 0 else None
            top_b = agg_b.sort_values('approved', ascending=False).iloc[0] if len(agg_b) > 0 else None
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            if top_a is not None:
                m1.metric(
                    f"Top Service — {month_a}",
                    top_a[loatype_col],
                    f"₱{top_a['approved']:,.0f}"
                )
            if top_b is not None:
                m2.metric(
                    f"Top Service — {month_b}",
                    top_b[loatype_col],
                    f"₱{top_b['approved']:,.0f}"
                )
            m3.metric(
                "MoM Change",
                f"{pct_change:+.1f}%",
                f"₱{(total_b - total_a):,.0f}"
            )
            
            st.divider()
            
            # Side-by-side comparison
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    agg_a.sort_values('approved', ascending=True),
                    x='approved',
                    y=loatype_col,
                    orientation='h',
                    title=f"{month_a} Service Mix",
                    labels={'approved': 'Total Approved (₱)', loatype_col: 'Service Type'},
                    color='approved',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    agg_b.sort_values('approved', ascending=True),
                    x='approved',
                    y=loatype_col,
                    orientation='h',
                    title=f"{month_b} Service Mix",
                    labels={'approved': 'Total Approved (₱)', loatype_col: 'Service Type'},
                    color='approved',
                    color_continuous_scale='Greens'
                )
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison table
            st.subheader("📊 Detailed Comparison")
            
            comparison = agg_a.merge(
                agg_b,
                on=loatype_col,
                how='outer',
                suffixes=('_A', '_B')
            ).fillna(0)
            
            comparison['Change (₱)'] = comparison['approved_B'] - comparison['approved_A']
            comparison['Change (%)'] = ((comparison['approved_B'] - comparison['approved_A']) / comparison['approved_A'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
            
            comparison = comparison.sort_values('Change (%)', ascending=False)
            comparison.columns = ['Service Type', f'{month_a} (₱)', f'{month_b} (₱)', 'Change (₱)', 'Change (%)']
            
            st.dataframe(
                comparison.style.format({
                    f'{month_a} (₱)': '₱{:,.0f}',
                    f'{month_b} (₱)': '₱{:,.0f}',
                    'Change (₱)': '₱{:,.0f}',
                    'Change (%)': '{:+.1f}%'
                }).background_gradient(subset=['Change (%)'], cmap='RdYlGn', vmin=-50, vmax=50),
                use_container_width=True
            )
        
        else:
            st.info("📊 Month column not detected — showing aggregated view")
        
        # Overall service mix
        st.divider()
        st.subheader("📊 Overall Service Mix (All Periods)")
        
        service_summary = df.groupby(loatype_col).agg({
            'approved': ['sum', 'mean', 'count']
        }).reset_index()
        service_summary.columns = ['Service Type', 'Total Approved', 'Avg Claim', 'Claim Count']
        service_summary = service_summary.sort_values('Total Approved', ascending=False)
        
        # Sunburst chart
        fig = px.sunburst(
            service_summary,
            path=['Service Type'],
            values='Total Approved',
            title='Service Type Hierarchy by Spend',
            color='Avg Claim',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.dataframe(
            service_summary.style.format({
                'Total Approved': '₱{:,.0f}',
                'Avg Claim': '₱{:,.0f}',
                'Claim Count': '{:,.0f}'
            }),
            use_container_width=True
        )
    
    # =====================================
    # PAGE 3: TOP UTILIZERS
    # =====================================
    elif page == "🧍‍♂️ Top Utilizers":
        st.markdown('<p class="main-header">🧍‍♂️ Top Utilizers Analysis</p>', unsafe_allow_html=True)
        
        possible_policy_cols = [
            c for c in df.columns
            if c.lower() in ["policyno", "cardnotext", "memberid", "member_no"]
        ]
        
        if not possible_policy_cols:
            st.error("❌ No member identifier column found (e.g., PolicyNo, CardNoText)")
            st.stop()
        
        policy_col = st.selectbox("Select Member Identifier", possible_policy_cols)
        
        # Aggregate by member
        utilizer_agg = df.groupby(policy_col).agg({
            'approved': ['sum', 'count', 'mean']
        }).reset_index()
        utilizer_agg.columns = [policy_col, 'Total_Approved', 'Claim_Count', 'Avg_Claim']
        utilizer_agg = utilizer_agg.sort_values('Total_Approved', ascending=False)
        
        # Top 1% analysis
        top_cutoff = np.percentile(utilizer_agg['Total_Approved'], 99)
        top_1p = utilizer_agg[utilizer_agg['Total_Approved'] >= top_cutoff]
        
        total_spend = utilizer_agg['Total_Approved'].sum()
        top_share = (top_1p['Total_Approved'].sum() / total_spend * 100) if total_spend > 0 else 0
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Top 1% Members", f"{len(top_1p):,}")
        col2.metric("Share of Total Spend", f"{top_share:.1f}%")
        col3.metric("Avg Spend per Top Member", f"₱{top_1p['Total_Approved'].mean():,.0f}")
        col4.metric("Avg Claims per Top Member", f"{top_1p['Claim_Count'].mean():.1f}")
        
        st.divider()
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Top 20 Members by Total Spend")
            
            top_20 = top_1p.head(20)
            fig = px.bar(
                top_20,
                x='Total_Approved',
                y=policy_col,
                orientation='h',
                title='Top 20 Highest Utilizers',
                labels={'Total_Approved': 'Total Approved (₱)', policy_col: 'Member ID'},
                color='Total_Approved',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=600, showlegend=False, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Spend Distribution")
            
            # Pareto chart
            utilizer_sorted = utilizer_agg.sort_values('Total_Approved', ascending=False).reset_index(drop=True)
            utilizer_sorted['Cumulative_Pct'] = (utilizer_sorted['Total_Approved'].cumsum() / utilizer_sorted['Total_Approved'].sum() * 100)
            utilizer_sorted['Member_Pct'] = ((utilizer_sorted.index + 1) / len(utilizer_sorted) * 100)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=utilizer_sorted['Member_Pct'],
                y=utilizer_sorted['Cumulative_Pct'],
                mode='lines',
                name='Cumulative Spend %',
                line=dict(color='#1f77b4', width=3)
            ))
            
            # Add 80-20 reference line
            fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% of Spend")
            fig.add_vline(x=20, line_dash="dash", line_color="red", annotation_text="20% of Members")
            
            fig.update_layout(
                title='Pareto Analysis: Member Concentration',
                xaxis_title='% of Members (ranked by spend)',
                yaxis_title='Cumulative % of Total Spend',
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table with demographics
        st.divider()
        st.subheader("📋 Top 1% Member Details")
        
        # Try to add demographic columns if available
        optional_cols = []
        for c in ['loatype', 'gender', 'general_tiering', 'providername']:
            match = [col for col in df.columns if col.lower() == c.lower()]
            if match:
                optional_cols.append(match[0])
        
        if optional_cols:
            demo_df = df[[policy_col] + optional_cols].drop_duplicates(subset=[policy_col])
            preview_df = top_1p.merge(demo_df, on=policy_col, how='left')
        else:
            preview_df = top_1p.copy()
        
        preview_df = preview_df.sort_values('Total_Approved', ascending=False)
        
        st.dataframe(
            preview_df.head(50).style.format({
                'Total_Approved': '₱{:,.0f}',
                'Avg_Claim': '₱{:,.0f}',
                'Claim_Count': '{:,.0f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Download option
        st.download_button(
            "📥 Download Top 1% Data",
            preview_df.to_csv(index=False).encode('utf-8'),
            "top_1_percent_utilizers.csv",
            "text/csv"
        )
        
        # Insights
        st.divider()
        st.markdown(f"""
        <div class="insight-box">
        <h4>💡 Utilization Concentration Insights</h4>
        <ul>
            <li><strong>High Concentration Risk:</strong> Top 1% ({len(top_1p):,} members) drive {top_share:.1f}% of total spend</li>
            <li><strong>Opportunity:</strong> Targeted care management for these members could yield 10-15% cost reduction (industry benchmark)</li>
            <li><strong>Recommendation:</strong> Implement high-touch case management, chronic disease programs, and care coordination initiatives</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # =====================================
    # PAGE 4: TREND ANALYSIS
    # =====================================
    elif page == "📈 Trend Analysis":
        st.markdown('<p class="main-header">📈 Trend Analysis</p>', unsafe_allow_html=True)
        
        month_cols = [c for c in df.columns if c.lower() in ['month', 'monthname', 'admissionmonth']]
        
        if not month_cols:
            st.error("❌ No month column found in the data. Trend analysis requires temporal data.")
            st.stop()
        
        month_col = month_cols[0]
        
        # Time series aggregation
        monthly_trend = df.groupby(month_col).agg({
            'approved': ['sum', 'count', 'mean']
        }).reset_index()
        monthly_trend.columns = ['Month', 'Total_Approved', 'Claim_Count', 'Avg_Claim']
        monthly_trend = monthly_trend.sort_values('Month')
        
        # Calculate growth rates
        monthly_trend['Approved_Growth'] = monthly_trend['Total_Approved'].pct_change() * 100
        monthly_trend['Volume_Growth'] = monthly_trend['Claim_Count'].pct_change() * 100
        
        # Metrics
        latest_month = monthly_trend.iloc[-1]
        previous_month = monthly_trend.iloc[-2] if len(monthly_trend) > 1 else monthly_trend.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Latest Month Spend",
            f"₱{latest_month['Total_Approved']:,.0f}",
            f"{latest_month['Approved_Growth']:+.1f}% MoM"
        )
        col2.metric(
            "Claim Volume",
            f"{latest_month['Claim_Count']:,.0f}",
            f"{latest_month['Volume_Growth']:+.1f}% MoM"
        )
        col3.metric(
            "Avg Claim Size",
            f"₱{latest_month['Avg_Claim']:,.0f}",
            f"₱{(latest_month['Avg_Claim'] - previous_month['Avg_Claim']):+,.0f}"
        )
        col4.metric(
            "Total Months",
            len(monthly_trend)
        )
        
        st.divider()
        
        # Main trend chart
        st.subheader("📊 Monthly Spend & Volume Trends")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Total Approved Amount', 'Claim Volume'),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Spend trend
        fig.add_trace(
            go.Scatter(
                x=monthly_trend['Month'],
                y=monthly_trend['Total_Approved'],
                mode='lines+markers',
                name='Total Approved',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Volume trend
        fig.add_trace(
            go.Scatter(
                x=monthly_trend['Month'],
                y=monthly_trend['Claim_Count'],
                mode='lines+markers',
                name='Claim Count',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Total Approved (₱)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Claims", row=2, col=1)
        
        fig.update_layout(height=700, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth rate chart
        st.subheader("📈 Month-over-Month Growth Rates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                monthly_trend,
                x='Month',
                y='Approved_Growth',
                title='Spend Growth Rate (%)',
                labels={'Approved_Growth': 'Growth Rate (%)'},
                color='Approved_Growth',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                monthly_trend,
                x='Month',
                y='Volume_Growth',
                title='Volume Growth Rate (%)',
                labels={'Volume_Growth': 'Growth Rate (%)'},
                color='Volume_Growth',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Service type trends
        st.divider()
        st.subheader("🧩 Service Type Trends")
        
        loatype_cols = [c for c in df.columns if 'loatype' in c.lower()]
        if loatype_cols:
            loatype_col = loatype_cols[0]
            
            service_trend = df.groupby([month_col, loatype_col])['approved'].sum().reset_index()
            
            fig = px.line(
                service_trend,
                x=month_col,
                y='approved',
                color=loatype_col,
                title='Service Type Spend Over Time',
                labels={'approved': 'Total Approved (₱)', month_col: 'Month'},
                markers=True
            )
            fig.update_layout(height=500, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # Market share evolution
            service_trend_pivot = service_trend.pivot(index=month_col, columns=loatype_col, values='approved').fillna(0)
            service_trend_pct = service_trend_pivot.div(service_trend_pivot.sum(axis=1), axis=0) * 100
            
            fig = go.Figure()
            
            for col in service_trend_pct.columns:
                fig.add_trace(go.Scatter(
                    x=service_trend_pct.index,
                    y=service_trend_pct[col],
                    mode='lines',
                    name=col,
                    stackgroup='one',
                    groupnorm='percent'
                ))
            
            fig.update_layout(
                title='Service Type Market Share Evolution',
                xaxis_title='Month',
                yaxis_title='Share of Total Spend (%)',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.divider()
        st.subheader("📋 Monthly Summary Table")
        
        st.dataframe(
            monthly_trend.style.format({
                'Total_Approved': '₱{:,.0f}',
                'Claim_Count': '{:,.0f}',
                'Avg_Claim': '₱{:,.0f}',
                'Approved_Growth': '{:+.1f}%',
                'Volume_Growth': '{:+.1f}%'
            }).background_gradient(subset=['Approved_Growth', 'Volume_Growth'], cmap='RdYlGn', vmin=-20, vmax=20),
            use_container_width=True
        )
    
    # =====================================
    # PAGE 5: PREDICTIVE INSIGHTS
    # =====================================
    elif page == "🎯 Predictive Insights":
        st.markdown('<p class="main-header">🎯 Predictive Insights</p>', unsafe_allow_html=True)
        
        # Anomaly Detection
        st.subheader("⚠️ Anomaly Detection")
        
        df_anomaly = detect_anomalies(df.copy())
        anomaly_count = df_anomaly['is_anomaly'].sum()
        anomaly_pct = (anomaly_count / len(df) * 100)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Anomalous Claims", f"{anomaly_count:,}")
        col2.metric("% of Total", f"{anomaly_pct:.2f}%")
        col3.metric("Total Anomaly Value", f"₱{df_anomaly[df_anomaly['is_anomaly']]['approved'].sum():,.0f}")
        
        # Anomaly visualization
        fig = px.scatter(
            df_anomaly,
            x=df_anomaly.index,
            y='approved',
            color='is_anomaly',
            title='Claims with Statistical Anomalies',
            labels={'approved': 'Approved Amount (₱)', 'index': 'Claim Index'},
            color_discrete_map={True: 'red', False: 'blue'}
        )
        
        mean_val = df['approved'].mean()
        threshold = mean_val + (3 * df['approved'].std())
        
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Anomaly Threshold: ₱{threshold:,.0f}")
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly details
        with st.expander("🔍 View Anomalous Claims"):
            anomalies = df_anomaly[df_anomaly['is_anomaly']].sort_values('approved', ascending=False)
            
            display_cols = ['approved'] + [c for c in anomalies.columns if c not in ['approved', 'is_anomaly']]
            display_cols = [c for c in display_cols if c in anomalies.columns][:10]
            
            st.dataframe(
                anomalies[display_cols].head(100),
                use_container_width=True,
                height=400
            )
            
            st.download_button(
                "📥 Download Anomalies",
                anomalies.to_csv(index=False).encode('utf-8'),
                "anomalous_claims.csv",
                "text/csv"
            )
        
        # Simple forecast
        st.divider()
        st.subheader("📊 Simple Trend Forecast")
        
        month_cols = [c for c in df.columns if c.lower() in ['month', 'monthname', 'admissionmonth']]
        
        if month_cols:
            month_col = month_cols[0]
            
            monthly_spend = df.groupby(month_col)['approved'].sum().reset_index()
            monthly_spend = monthly_spend.sort_values(month_col)
            monthly_spend['month_idx'] = range(len(monthly_spend))
            
            # Simple linear regression
            from numpy.polynomial import Polynomial
            
            p = Polynomial.fit(monthly_spend['month_idx'], monthly_spend['approved'], 1)
            
            # Forecast next 3 months
            future_months = len(monthly_spend) + np.arange(1, 4)
            forecast_values = p(future_months)
            
            # Combine historical and forecast
            forecast_df = pd.DataFrame({
                'Month': [f'Forecast +{i}' for i in range(1, 4)],
                'month_idx': future_months,
                'Approved': forecast_values,
                'Type': 'Forecast'
            })
            
            monthly_spend['Type'] = 'Historical'
            monthly_spend = monthly_spend.rename(columns={'approved': 'Approved', month_col: 'Month'})
            
            combined = pd.concat([
                monthly_spend[['Month', 'month_idx', 'Approved', 'Type']],
                forecast_df
            ])
            
            fig = px.line(
                combined,
                x='month_idx',
                y='Approved',
                color='Type',
                markers=True,
                title='Spend Forecast (Linear Trend)',
                labels={'Approved': 'Total Approved (₱)', 'month_idx': 'Time Period'},
                color_discrete_map={'Historical': 'blue', 'Forecast': 'red'}
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            col1, col2, col3 = st.columns(3)
            col1.metric("Next Month Forecast", f"₱{forecast_values[0]:,.0f}")
            col2.metric("2 Months Ahead", f"₱{forecast_values[1]:,.0f}")
            col3.metric("3 Months Ahead", f"₱{forecast_values[2]:,.0f}")
            
            st.info("📝 Note: This is a simple linear trend forecast. For production use, consider advanced time series models (ARIMA, Prophet, etc.)")
        
        else:
            st.warning("⚠️ Time series forecasting requires a month column in your data")
        
        # Risk Stratification
        st.divider()
        st.subheader("🎯 Risk Stratification")
        
        # Create risk segments based on approved amounts
        df_risk = df.copy()
        
        q25 = df_risk['approved'].quantile(0.25)
        q75 = df_risk['approved'].quantile(0.75)
        q95 = df_risk['approved'].quantile(0.95)
        
        def risk_category(amount):
            if amount <= q25:
                return 'Low Risk'
            elif amount <= q75:
                return 'Medium Risk'
            elif amount <= q95:
                return 'High Risk'
            else:
                return 'Critical Risk'
        
        df_risk['Risk_Category'] = df_risk['approved'].apply(risk_category)
        
        risk_summary = df_risk.groupby('Risk_Category').agg({
            'approved': ['sum', 'count', 'mean']
        }).reset_index()
        risk_summary.columns = ['Risk Category', 'Total Spend', 'Claim Count', 'Avg Claim']
        
        # Define order
        risk_order = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        risk_summary['Risk Category'] = pd.Categorical(risk_summary['Risk Category'], categories=risk_order, ordered=True)
        risk_summary = risk_summary.sort_values('Risk Category')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                risk_summary,
                values='Claim Count',
                names='Risk Category',
                title='Claims by Risk Category',
                color='Risk Category',
                color_discrete_map={
                    'Low Risk': '#90EE90',
                    'Medium Risk': '#FFD700',
                    'High Risk': '#FFA500',
                    'Critical Risk': '#FF4500'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                risk_summary,
                values='Total Spend',
                names='Risk Category',
                title='Spend by Risk Category',
                color='Risk Category',
                color_discrete_map={
                    'Low Risk': '#90EE90',
                    'Medium Risk': '#FFD700',
                    'High Risk': '#FFA500',
                    'Critical Risk': '#FF4500'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            risk_summary.style.format({
                'Total Spend': '₱{:,.0f}',
                'Claim Count': '{:,.0f}',
                'Avg Claim': '₱{:,.0f}'
            }),
            use_container_width=True
        )
        
        # Insights
        critical_spend = risk_summary[risk_summary['Risk Category'] == 'Critical Risk']['Total Spend'].values[0] if len(risk_summary[risk_summary['Risk Category'] == 'Critical Risk']) > 0 else 0
        critical_pct = (critical_spend / df['approved'].sum() * 100) if df['approved'].sum() > 0 else 0
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>💡 Risk-Based Insights</h4>
        <ul>
            <li><strong>Critical Risk Claims:</strong> Account for {critical_pct:.1f}% of total spend</li>
            <li><strong>Recommendation:</strong> Implement pre-authorization for high-risk procedures</li>
            <li><strong>Focus Area:</strong> Enhanced utilization review for claims above ₱{q95:,.0f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # =====================================
    # PAGE 6: DATA EXPLORER
    # =====================================
    elif page == "🔍 Data Explorer":
        st.markdown('<p class="main-header">🔍 Data Explorer</p>', unsafe_allow_html=True)
        
        # Data quality metrics
        st.subheader("📊 Data Quality Report")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column profiling
        st.divider()
        st.subheader("📋 Column Profiling")
        
        col_stats = []
        for col in df.columns:
            col_stats.append({
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-Null': df[col].notna().sum(),
                'Null': df[col].isna().sum(),
                'Unique': df[col].nunique(),
                'Sample': str(df[col].iloc[0]) if len(df) > 0 else 'N/A'
            })
        
        col_stats_df = pd.DataFrame(col_stats)
        st.dataframe(col_stats_df, use_container_width=True, height=400)
        
        # Interactive data table
        st.divider()
        st.subheader("📄 Interactive Data Table")
        
        # Column selector
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to display",
            options=all_columns,
            default=all_columns[:10]
        )
        
        if selected_columns:
            # Search functionality
            search_term = st.text_input("🔍 Search in data", "")
            
            display_df = df[selected_columns].copy()
            
            if search_term:
                # Search across all string columns
                mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                display_df = display_df[mask]
            
            st.info(f"Showing {len(display_df):,} records")
            
            # Editable dataframe
            st.data_editor(
                display_df.head(100),
                use_container_width=True,
                height=400,
                num_rows="fixed"
            )
            
            # Download options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Filtered Data (CSV)",
                    display_df.to_csv(index=False).encode('utf-8'),
                    "filtered_data.csv",
                    "text/csv"
                )
            with col2:
                st.download_button(
                    "📥 Download Full Dataset (CSV)",
                    df.to_csv(index=False).encode('utf-8'),
                    "full_dataset.csv",
                    "text/csv"
                )
        
        # Statistical summary
        st.divider()
        st.subheader("📊 Statistical Summary")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_numeric = st.selectbox("Select numeric column for analysis", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution
                fig = px.histogram(
                    df,
                    x=selected_numeric,
                    title=f'Distribution of {selected_numeric}',
                    nbins=50
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(
                    df,
                    y=selected_numeric,
                    title=f'Box Plot of {selected_numeric}'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Descriptive statistics
            stats_df = df[selected_numeric].describe().to_frame()
            stats_df.columns = ['Value']
            st.dataframe(stats_df.style.format({'Value': '{:,.2f}'}), use_container_width=True)