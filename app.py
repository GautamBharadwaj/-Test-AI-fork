import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Page configuration
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
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🚀 Dashboard Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Overview", "📈 Data Analysis", "🤖 ML Predictions", "📊 Interactive Charts", "🌐 API Data", "⚙️ Settings"]
)

# Generate sample data
@st.cache_data
def generate_sample_data():
    """Generate sample datasets for the dashboard"""
    np.random.seed(42)
    
    # Sales data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    sales_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 100,
        'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
        'product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], len(dates))
    })
    sales_data['sales'] = np.maximum(sales_data['sales'], 0)  # Ensure positive values
    
    # User engagement data
    user_data = pd.DataFrame({
        'user_id': range(1, 1001),
        'age': np.random.randint(18, 65, 1000),
        'session_duration': np.random.exponential(15, 1000),
        'pages_visited': np.random.poisson(5, 1000),
        'conversion': np.random.choice([0, 1], 1000, p=[0.85, 0.15])
    })
    
    return sales_data, user_data

# Load data
sales_data, user_data = generate_sample_data()

# Main content based on selected page
if page == "🏠 Overview":
    st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("### Welcome to your comprehensive analytics dashboard!")
    st.markdown("Use the sidebar to navigate between different sections and explore various data insights.")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = sales_data['sales'].sum()
        st.metric(
            label="💰 Total Sales",
            value=f"${total_sales:,.0f}",
            delta=f"{np.random.uniform(-5, 15):.1f}%"
        )
    
    with col2:
        avg_session = user_data['session_duration'].mean()
        st.metric(
            label="⏱️ Avg Session Duration",
            value=f"{avg_session:.1f} min",
            delta=f"{np.random.uniform(-2, 8):.1f}%"
        )
    
    with col3:
        conversion_rate = user_data['conversion'].mean() * 100
        st.metric(
            label="🎯 Conversion Rate",
            value=f"{conversion_rate:.1f}%",
            delta=f"{np.random.uniform(-1, 3):.1f}%"
        )
    
    with col4:
        total_users = len(user_data)
        st.metric(
            label="👥 Total Users",
            value=f"{total_users:,}",
            delta=f"{np.random.randint(10, 50)}"
        )
    
    # Overview charts
    st.markdown("### 📈 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales trend
        monthly_sales = sales_data.groupby(sales_data['date'].dt.to_period('M'))['sales'].sum()
        fig_sales = px.line(
            x=monthly_sales.index.astype(str), 
            y=monthly_sales.values,
            title="Monthly Sales Trend",
            labels={'x': 'Month', 'y': 'Sales ($)'}
        )
        fig_sales.update_layout(showlegend=False)
        st.plotly_chart(fig_sales, use_container_width=True)
    
    with col2:
        # User age distribution
        fig_age = px.histogram(
            user_data, 
            x='age', 
            nbins=20,
            title="User Age Distribution",
            labels={'age': 'Age', 'count': 'Number of Users'}
        )
        st.plotly_chart(fig_age, use_container_width=True)

elif page == "📈 Data Analysis":
    st.markdown('<h1 class="main-header">📈 Data Analysis</h1>', unsafe_allow_html=True)
    
    # Data selection
    st.sidebar.markdown("### 🔧 Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Choose analysis type:",
        ["Sales Analysis", "User Behavior", "Regional Performance"]
    )
    
    if analysis_type == "Sales Analysis":
        st.markdown("### 💰 Sales Performance Analysis")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=sales_data['date'].min())
        with col2:
            end_date = st.date_input("End Date", value=sales_data['date'].max())
        
        # Filter data
        mask = (sales_data['date'].dt.date >= start_date) & (sales_data['date'].dt.date <= end_date)
        filtered_data = sales_data.loc[mask]
        
        # Sales by product
        product_sales = filtered_data.groupby('product')['sales'].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_product = px.bar(
                x=product_sales.index,
                y=product_sales.values,
                title="Sales by Product",
                labels={'x': 'Product', 'y': 'Total Sales ($)'}
            )
            st.plotly_chart(fig_product, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                values=product_sales.values,
                names=product_sales.index,
                title="Sales Distribution by Product"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Sales heatmap by region and product
        heatmap_data = filtered_data.groupby(['region', 'product'])['sales'].sum().unstack()
        fig_heatmap = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            title="Sales Heatmap: Region vs Product",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    elif analysis_type == "User Behavior":
        st.markdown("### 👥 User Behavior Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Session duration vs conversion
            fig_scatter = px.scatter(
                user_data,
                x='session_duration',
                y='pages_visited',
                color='conversion',
                title="Session Duration vs Pages Visited",
                labels={'session_duration': 'Session Duration (min)', 'pages_visited': 'Pages Visited'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Age group analysis
            user_data['age_group'] = pd.cut(user_data['age'], bins=[18, 25, 35, 45, 55, 65], labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
            age_conversion = user_data.groupby('age_group')['conversion'].mean()
            
            fig_age_conv = px.bar(
                x=age_conversion.index,
                y=age_conversion.values,
                title="Conversion Rate by Age Group",
                labels={'x': 'Age Group', 'y': 'Conversion Rate'}
            )
            st.plotly_chart(fig_age_conv, use_container_width=True)
        
        # Correlation matrix
        st.markdown("### 🔗 Feature Correlations")
        corr_matrix = user_data[['age', 'session_duration', 'pages_visited', 'conversion']].corr()
        fig_corr = px.imshow(
            corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    elif analysis_type == "Regional Performance":
        st.markdown("### 🌍 Regional Performance Analysis")
        
        regional_stats = sales_data.groupby('region').agg({
            'sales': ['sum', 'mean', 'count']
        }).round(2)
        regional_stats.columns = ['Total Sales', 'Average Sales', 'Number of Transactions']
        
        st.dataframe(regional_stats, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_regional = px.bar(
                x=regional_stats.index,
                y=regional_stats['Total Sales'],
                title="Total Sales by Region",
                labels={'x': 'Region', 'y': 'Total Sales ($)'}
            )
            st.plotly_chart(fig_regional, use_container_width=True)
        
        with col2:
            fig_avg = px.bar(
                x=regional_stats.index,
                y=regional_stats['Average Sales'],
                title="Average Sales by Region",
                labels={'x': 'Region', 'y': 'Average Sales ($)'}
            )
            st.plotly_chart(fig_avg, use_container_width=True)

elif page == "🤖 ML Predictions":
    st.markdown('<h1 class="main-header">🤖 Machine Learning Predictions</h1>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 User Conversion Prediction")
    
    # Prepare ML data
    X = user_data[['age', 'session_duration', 'pages_visited']]
    y = user_data['conversion']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Model Accuracy", f"{accuracy:.3f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig_importance = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title="Feature Importance"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.markdown("### 🔮 Make a Prediction")
        
        # User input for prediction
        pred_age = st.slider("Age", min_value=18, max_value=65, value=30)
        pred_duration = st.slider("Session Duration (min)", min_value=0.0, max_value=60.0, value=15.0)
        pred_pages = st.slider("Pages Visited", min_value=1, max_value=20, value=5)
        
        if st.button("🎯 Predict Conversion"):
            prediction = model.predict([[pred_age, pred_duration, pred_pages]])[0]
            probability = model.predict_proba([[pred_age, pred_duration, pred_pages]])[0][1]
            
            if prediction == 1:
                st.success(f"✅ High conversion probability: {probability:.2%}")
            else:
                st.warning(f"⚠️ Low conversion probability: {probability:.2%}")
    
    # Classification report
    st.markdown("### 📋 Detailed Model Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

elif page == "📊 Interactive Charts":
    st.markdown('<h1 class="main-header">📊 Interactive Charts</h1>', unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🎛️ Chart Controls")
    
    chart_type = st.sidebar.selectbox(
        "Select chart type:",
        ["Line Chart", "Scatter Plot", "Box Plot", "Violin Plot", "3D Scatter"]
    )
    
    if chart_type == "Line Chart":
        st.markdown("### 📈 Time Series Analysis")
        
        # Multi-select for products
        selected_products = st.multiselect(
            "Select products to display:",
            sales_data['product'].unique(),
            default=sales_data['product'].unique()[:2]
        )
        
        if selected_products:
            filtered_data = sales_data[sales_data['product'].isin(selected_products)]
            daily_sales = filtered_data.groupby(['date', 'product'])['sales'].sum().reset_index()
            
            fig = px.line(
                daily_sales,
                x='date',
                y='sales',
                color='product',
                title="Daily Sales by Product"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Scatter Plot":
        st.markdown("### 🎯 Scatter Plot Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis:", ['age', 'session_duration', 'pages_visited'])
        with col2:
            y_axis = st.selectbox("Y-axis:", ['session_duration', 'pages_visited', 'age'])
        
        if x_axis != y_axis:
            fig = px.scatter(
                user_data,
                x=x_axis,
                y=y_axis,
                color='conversion',
                size='session_duration',
                title=f"{x_axis.title()} vs {y_axis.title()}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Box Plot":
        st.markdown("### 📦 Distribution Analysis")
        
        metric = st.selectbox("Select metric:", ['sales', 'session_duration', 'pages_visited'])
        
        if metric == 'sales':
            fig = px.box(sales_data, x='region', y='sales', title="Sales Distribution by Region")
        else:
            fig = px.box(user_data, x='conversion', y=metric, title=f"{metric.title()} Distribution by Conversion")
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Violin Plot":
        st.markdown("### 🎻 Density Distribution")
        
        fig = px.violin(
            user_data,
            x='conversion',
            y='session_duration',
            title="Session Duration Distribution by Conversion Status"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "3D Scatter":
        st.markdown("### 🌐 3D Visualization")
        
        fig = px.scatter_3d(
            user_data.sample(200),  # Sample for performance
            x='age',
            y='session_duration',
            z='pages_visited',
            color='conversion',
            title="3D User Behavior Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🌐 API Data":
    st.markdown('<h1 class="main-header">🌐 Live API Data</h1>', unsafe_allow_html=True)
    
    st.markdown("### 🔄 Real-time Data Integration")
    
    # API data simulation (replace with real API calls)
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_api_data():
        """Simulate API data fetching"""
        # In a real scenario, you'd make actual API calls here
        return {
            'weather': {
                'temperature': np.random.uniform(15, 30),
                'humidity': np.random.uniform(30, 80),
                'pressure': np.random.uniform(1000, 1020)
            },
            'stock_prices': pd.DataFrame({
                'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
                'price': np.random.uniform(100, 300, 5),
                'change': np.random.uniform(-5, 5, 5)
            })
        }
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
    
    api_data = fetch_api_data()
    
    # Weather data
    st.markdown("### 🌤️ Weather Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌡️ Temperature", f"{api_data['weather']['temperature']:.1f}°C")
    with col2:
        st.metric("💧 Humidity", f"{api_data['weather']['humidity']:.1f}%")
    with col3:
        st.metric("📊 Pressure", f"{api_data['weather']['pressure']:.0f} hPa")
    
    # Stock data
    st.markdown("### 📈 Stock Prices")
    stock_df = api_data['stock_prices']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_stocks = px.bar(
            stock_df,
            x='symbol',
            y='price',
            title="Current Stock Prices",
            color='change',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_stocks, use_container_width=True)
    
    with col2:
        # Display stock table
        st.dataframe(
            stock_df.style.format({'price': '${:.2f}', 'change': '{:+.2f}%'}),
            use_container_width=True
        )

elif page == "⚙️ Settings":
    st.markdown('<h1 class="main-header">⚙️ Dashboard Settings</h1>', unsafe_allow_html=True)
    
    st.markdown("### 🎨 Customization Options")
    
    # Theme selection
    theme = st.selectbox(
        "Choose dashboard theme:",
        ["Default", "Dark", "Light", "Custom"]
    )
    
    # Data refresh settings
    st.markdown("### 🔄 Data Settings")
    auto_refresh = st.checkbox("Enable auto-refresh", value=False)
    
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", min_value=30, max_value=300, value=60)
        st.info(f"Data will refresh every {refresh_interval} seconds")
    
    # Export options
    st.markdown("### 📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Sales Data"):
            csv = sales_data.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="sales_data.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("👥 Export User Data"):
            csv = user_data.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="user_data.csv",
                mime="text/csv"
            )
    
    # System info
    st.markdown("### ℹ️ System Information")
    st.info(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.info(f"Total data points: {len(sales_data)} sales records, {len(user_data)} user records")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        📊 Analytics Dashboard | Built with Streamlit | Last updated: {}
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
    unsafe_allow_html=True
)