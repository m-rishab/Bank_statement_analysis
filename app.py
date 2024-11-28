import streamlit as st
st.set_page_config(layout='wide', page_title='Advanced Bank Transaction Analysis')

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from scipy import stats

class AdvancedBankAnalysisDashboard:
    def __init__(self, data):
        """Initialize the dashboard with bank transaction data"""
        self.data = self._preprocess_data(data)
        self.credits = self.data[self.data['DrCr'] == 'Cr']
        self.debits = self.data[self.data['DrCr'] == 'Db']

    def _preprocess_data(self, data):
        """Advanced data preprocessing"""
        data['date'] = pd.to_datetime(data['date'])
        data['name'] = data['name'].fillna('Unknown')
        data['month'] = data['date'].dt.to_period('M')
        data['day_of_week'] = data['date'].dt.day_name()
        data['quarter'] = data['date'].dt.to_period('Q')
        return data.drop_duplicates()

    def calculate_advanced_metrics(self):
        """Calculate comprehensive financial metrics"""
        total_income = self.credits['amount'].sum()
        total_expenses = self.debits['amount'].sum()
        savings_rate = (total_income - total_expenses) / total_income * 100
        
        # Cash flow volatility
        monthly_cash_flow = self.data.groupby(self.data['date'].dt.to_period('M'))['amount'].sum()
        cash_flow_volatility = monthly_cash_flow.std() / monthly_cash_flow.mean() * 100

        return {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'savings_rate': savings_rate,
            'avg_monthly_income': self.credits.groupby('month')['amount'].sum().mean(),
            'avg_monthly_expenses': self.debits.groupby('month')['amount'].sum().mean(),
            'cash_flow_volatility': cash_flow_volatility
        }

    def detect_anomalies_advanced(self):
        """Advanced anomaly detection using multiple methods"""
        # Z-score method
        z_score_anomalies = self._detect_zscore_anomalies()
        
        # Interquartile range method
        iqr_anomalies = self._detect_iqr_anomalies()
        
        # Time-based unusual transactions
        time_anomalies = self._detect_time_based_anomalies()
        
        return {
            'z_score_anomalies': z_score_anomalies,
            'iqr_anomalies': iqr_anomalies,
            'time_anomalies': time_anomalies
        }

    def _detect_zscore_anomalies(self):
        """Detect anomalies using Z-score method"""
        z_scores = np.abs(stats.zscore(self.data['amount']))
        return self.data[z_scores > 3]

    def _detect_iqr_anomalies(self):
        """Detect anomalies using Interquartile Range method"""
        Q1 = self.data['amount'].quantile(0.25)
        Q3 = self.data['amount'].quantile(0.75)
        IQR = Q3 - Q1
        return self.data[(self.data['amount'] < (Q1 - 1.5 * IQR)) | (self.data['amount'] > (Q3 + 1.5 * IQR))]

    def _detect_time_based_anomalies(self):
        """Detect unusual time-based transactions"""
        # Find transactions outside typical transaction hours or days
        late_night_transactions = self.data[
            (self.data['date'].dt.hour < 6) | (self.data['date'].dt.hour > 22)
        ]
        weekend_transactions = self.data[self.data['date'].dt.day_name().isin(['Saturday', 'Sunday'])]
        
        return {
            'late_night_transactions': late_night_transactions,
            'weekend_transactions': weekend_transactions
        }

    def visualize_advanced_trends(self):
        """Create advanced trend visualizations"""
        # Monthly income vs expenses with trend line
        monthly_summary = self.data.groupby([self.data['date'].dt.to_period('M'), 'DrCr'])['amount'].sum().unstack()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_summary.index.astype(str), 
            y=monthly_summary['Cr'], 
            mode='lines+markers', 
            name='Income', 
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=monthly_summary.index.astype(str), 
            y=monthly_summary['Db'], 
            mode='lines+markers', 
            name='Expenses', 
            line=dict(color='red')
        ))
        
        # Trend line
        trend_line = np.polyfit(range(len(monthly_summary)), monthly_summary['Db'], 1)
        trend_fn = np.poly1d(trend_line)
        
        fig.add_trace(go.Scatter(
            x=monthly_summary.index.astype(str), 
            y=trend_fn(range(len(monthly_summary))), 
            mode='lines', 
            name='Expense Trend', 
            line=dict(color='purple', dash='dot')
        ))
        
        fig.update_layout(
            title='Advanced Monthly Income vs Expenses Analysis (₹)',
            xaxis_title='Month',
            yaxis_title='Amount (₹)',
            height=500
        )
        
        return fig

    def visualize_comprehensive_distribution(self):
        """Create comprehensive transaction distribution visualizations"""
        # Spending by day of week
        day_spending = self.debits.groupby('day_of_week')['amount'].sum()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_spending = day_spending.reindex(day_order)
        
        fig_day_spend = px.bar(
            x=day_spending.index, 
            y=day_spending.values, 
            title='Spending Distribution by Day of Week (₹)',
            labels={'x': 'Day', 'y': 'Total Amount (₹)'}
        )
        
        # Advanced category treemap
        category_spending = self.debits.groupby('name')['amount'].sum().nlargest(15)
        fig_category_treemap = px.treemap(
            names=category_spending.index, 
            values=category_spending.values, 
            title='Top 15 Spending Categories Treemap (₹)'
        )
        
        return fig_day_spend, fig_category_treemap

    def visualize_additional_insights(self):
        """Create additional advanced visualizations"""
        # 1. Cumulative Cash Flow Chart
        cumulative_cash_flow = self.data.groupby('date')['amount'].sum().cumsum()
        fig_cash_flow = go.Figure()
        fig_cash_flow.add_trace(go.Scatter(
            x=cumulative_cash_flow.index, 
            y=cumulative_cash_flow.values, 
            mode='lines', 
            name='Cumulative Cash Flow',
            line=dict(color='blue', width=2)
        ))
        fig_cash_flow.update_layout(
            title='Cumulative Cash Flow Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Amount (₹)',
            height=400
        )

        # 2. Transaction Frequency Heatmap
        transaction_freq = self.data.groupby([
            self.data['date'].dt.day_name(), 
            self.data['date'].dt.hour
        ])['amount'].count().unstack()
        
        fig_heatmap = px.imshow(
            transaction_freq, 
            labels=dict(x='Hour of Day', y='Day of Week', color='Transaction Count'),
            title='Transaction Frequency Heatmap'
        )

        # 3. Recurring Transactions Analysis
        recurring_threshold = self.data['amount'].mean()
        recurring_transactions = self.data[
            self.data.groupby('name')['amount'].transform('mean') > recurring_threshold
        ]
        recurring_summary = recurring_transactions.groupby('name')['amount'].agg(['mean', 'count'])
        
        fig_recurring = px.scatter(
            recurring_summary, 
            x='count', 
            y='mean', 
            size='count',
            hover_name=recurring_summary.index,
            title='Recurring Transactions Analysis',
            labels={'count': 'Frequency', 'mean': 'Average Amount'}
        )

        return fig_cash_flow, fig_heatmap, fig_recurring

    def generate_advanced_recommendations(self, metrics, anomalies):
        """Generate comprehensive financial recommendations"""
        recommendations = []
        
        # Savings and expense recommendations
        if metrics['savings_rate'] < 0:
            recommendations.append("🚨 Critical: Your expenses exceed income. Urgent need to reduce spending.")
        elif metrics['savings_rate'] < 10:
            recommendations.append("⚠️ Low savings rate. Consider optimizing expenses.")
        
        # Cash flow volatility recommendations
        if metrics['cash_flow_volatility'] > 50:
            recommendations.append("💡 High cash flow variability. Consider building an emergency fund.")
        
        # Anomaly-based recommendations
        z_score_count = len(anomalies['z_score_anomalies'])
        iqr_count = len(anomalies['iqr_anomalies'])
        
        if z_score_count > 0:
            recommendations.append(f"🔍 Investigate {z_score_count} statistically unusual transactions.")
        
        # Time-based anomaly insights
        late_night_count = len(anomalies['time_anomalies']['late_night_transactions'])
        weekend_count = len(anomalies['time_anomalies']['weekend_transactions'])
        
        if late_night_count > 0:
            recommendations.append(f"🌙 {late_night_count} late-night transactions detected. Potential security concern.")
        
        if weekend_count > 0:
            recommendations.append(f"🗓️ {weekend_count} weekend transactions found. Review unusual spending patterns.")
        
        return recommendations

def render_advanced_dashboard(data):
    """Advanced Streamlit dashboard rendering function"""
    # Initialize analysis
    analyzer = AdvancedBankAnalysisDashboard(data)
    metrics = analyzer.calculate_advanced_metrics()
    anomalies = analyzer.detect_anomalies_advanced()
    
    # Dashboard Layout
    st.title('💰 Bank Statement Analysis Dashboard')
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Income", f"₹{metrics['total_income']:,.2f}")
    col2.metric("Total Expenses", f"₹{metrics['total_expenses']:,.2f}")
    col3.metric("Savings Rate", f"{metrics['savings_rate']:.2f}%")
    col4.metric("Cash Flow Volatility", f"{metrics['cash_flow_volatility']:.2f}%")
    
    # Visualization Rows
    st.plotly_chart(analyzer.visualize_advanced_trends(), use_container_width=True)
    
    # Distribution Visualizations
    col_day, col_category = st.columns(2)
    
    with col_day:
        st.plotly_chart(analyzer.visualize_comprehensive_distribution()[0])
    
    with col_category:
        st.plotly_chart(analyzer.visualize_comprehensive_distribution()[1])
    
    # New Additional Visualizations Section
    st.header('🔍 Advanced Financial Insights')
    
    # Create columns for new visualizations
    col_cash_flow, col_freq, col_recurring = st.columns(3)
    
    # Additional Visualizations
    cash_flow_fig, heatmap_fig, recurring_fig = analyzer.visualize_additional_insights()
    
    with col_cash_flow:
        st.plotly_chart(cash_flow_fig, use_container_width=True)
    
    with col_freq:
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    with col_recurring:
        st.plotly_chart(recurring_fig, use_container_width=True)
    
    # Anomalies Section
    st.header('🚨 Anomaly Insights')
    st.dataframe(anomalies['z_score_anomalies'])
    
    # Recommendations Section
    st.header('🎯 Smart Financial Recommendations')
    recommendations = analyzer.generate_advanced_recommendations(metrics, anomalies)
    for rec in recommendations:
        st.markdown(f"- {rec}")

def main():
    # Directly load the local CSV file
    data_path = 'Data/bankstatements.csv'
    try:
        data = pd.read_csv(data_path)
        render_advanced_dashboard(data)
    except Exception as e:
        st.error(f"Error loading file: {e}")

if __name__ == '__main__':
    main()