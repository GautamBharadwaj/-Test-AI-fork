import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Set page config
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample data for the dashboard"""
    # Generate date range
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Sales data
    sales_data = []
    for date in dates:
        daily_sales = random.randint(1000, 5000)
        sales_data.append({
            'date': date,
            'sales': daily_sales,
            'month': date.strftime('%B'),
            'quarter': f"Q{(date.month-1)//3 + 1}"
        })
    
    # User engagement data
    user_data = {
        'metric': ['Active Users', 'New Signups', 'Page Views', 'Session Duration (min)'],
        'current': [15420, 1250, 45680, 12.5],
        'previous': [14800, 1180, 42300, 11.8],
        'change': [4.2, 5.9, 8.0, 5.9]
    }
    
    # Regional data
    regional_data = {
        'region': ['North America', 'Europe', 'Asia Pacific', 'South America', 'Africa'],
        'sales': [45000, 38000, 52000, 15000, 8000],
        'users': [12000, 9500, 15000, 4500, 2000]
    }
    
    return pd.DataFrame(sales_data), pd.DataFrame(user_data), pd.DataFrame(regional_data)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
        min_value=datetime(2024, 1, 1),
        max_value=datetime(2024, 12, 31)
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Generate data
    sales_df, user_df, regional_df = generate_sample_data()
    
    # Filter data based on date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        sales_df = sales_df[(sales_df['date'] >= pd.Timestamp(start_date)) & 
                           (sales_df['date'] <= pd.Timestamp(end_date))]
    
    # Key Metrics Row
    st.markdown("## 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = sales_df['sales'].sum()
        st.metric(
            label="Total Sales",
            value=f"${total_sales:,}",
            delta=f"+{random.randint(5, 15)}%"
        )
    
    with col2:
        avg_daily_sales = sales_df['sales'].mean()
        st.metric(
            label="Avg Daily Sales",
            value=f"${avg_daily_sales:,.0f}",
            delta=f"+{random.randint(2, 8)}%"
        )
    
    with col3:
        active_users = user_df[user_df['metric'] == 'Active Users']['current'].iloc[0]
        user_change = user_df[user_df['metric'] == 'Active Users']['change'].iloc[0]
        st.metric(
            label="Active Users",
            value=f"{active_users:,}",
            delta=f"+{user_change}%"
        )
    
    with col4:
        total_regions = len(regional_df)
        st.metric(
            label="Active Regions",
            value=total_regions,
            delta="+1"
        )
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Sales Trend")
        # Sales trend chart
        fig_sales = px.line(
            sales_df, 
            x='date', 
            y='sales',
            title='Daily Sales Over Time',
            template='plotly_white'
        )
        fig_sales.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            height=400
        )
        st.plotly_chart(fig_sales, use_container_width=True)
    
    with col2:
        st.markdown("### 🌍 Regional Performance")
        # Regional sales chart
        fig_regional = px.bar(
            regional_df,
            x='region',
            y='sales',
            title='Sales by Region',
            template='plotly_white',
            color='sales',
            color_continuous_scale='Blues'
        )
        fig_regional.update_layout(
            xaxis_title="Region",
            yaxis_title="Sales ($)",
            height=400
        )
        st.plotly_chart(fig_regional, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Monthly Sales Distribution")
        monthly_sales = sales_df.groupby('month')['sales'].sum().reset_index()
        fig_monthly = px.pie(
            monthly_sales,
            values='sales',
            names='month',
            title='Sales Distribution by Month',
            template='plotly_white'
        )
        fig_monthly.update_layout(height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        st.markdown("### 👥 User Metrics")
        # User metrics comparison
        fig_users = go.Figure()
        
        fig_users.add_trace(go.Bar(
            name='Current',
            x=user_df['metric'],
            y=user_df['current'],
            marker_color='lightblue'
        ))
        
        fig_users.add_trace(go.Bar(
            name='Previous',
            x=user_df['metric'],
            y=user_df['previous'],
            marker_color='lightcoral'
        ))
        
        fig_users.update_layout(
            title='Current vs Previous Period',
            xaxis_title="Metrics",
            yaxis_title="Values",
            barmode='group',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_users, use_container_width=True)
    
    # Data Tables Section
    st.markdown("---")
    st.markdown("## 📋 Data Tables")
    
    tab1, tab2, tab3 = st.tabs(["Sales Data", "User Metrics", "Regional Data"])
    
    with tab1:
        st.markdown("### Recent Sales Data")
        st.dataframe(
            sales_df.tail(10)[['date', 'sales', 'month']].sort_values('date', ascending=False),
            use_container_width=True
        )
    
    with tab2:
        st.markdown("### User Engagement Metrics")
        st.dataframe(user_df, use_container_width=True)
    
    with tab3:
        st.markdown("### Regional Performance")
        st.dataframe(regional_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Dashboard last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()