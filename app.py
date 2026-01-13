"""
Streamlit interface for Credit Risk Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

# Load the saved model and preprocessors
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')

def preprocess_input(user_input):
    """Preprocess user input using saved transformers"""
    processed_input = {}
    
    for feature in feature_names:
        if feature in label_encoders:
            processed_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
        else:
            processed_input[feature] = user_input[feature]
    
    # Convert to DataFrame and scale
    input_df = pd.DataFrame([processed_input])
    input_scaled = scaler.transform(input_df)
    return input_scaled

def main():
    st.title("Credit Risk Assessment System")
    st.write("Enter customer information to assess credit risk")
    
    with st.form("credit_risk_form"):
        # Demographics
        st.subheader("Demographics")
        age = st.slider("Age", 21, 65, 35)
        occupation = st.selectbox("Occupation", 
                                ["Salaried", "Self-employed", "Business", "Freelancer", "Professional"])
        annual_income = st.number_input("Annual Income (₹)", 
                                      min_value=15000, 
                                      max_value=300000, 
                                      value=50000)
        city = st.selectbox("City",
                           ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 
                            'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad'])
        education = st.selectbox("Education",
                               ['Graduate', 'Post-Graduate', 'Undergraduate', 'Doctorate'])
        
        # Employment and Credit History
        st.subheader("Employment and Credit History")
        employment_length = st.slider("Employment Length (years)", 0, 40, 5)
        account_age = st.slider("Credit History Length (months)", 6, 240, 24)
        existing_loans = st.number_input("Number of Existing Loans", 0, 5, 1)
        
        # Transaction Patterns
        st.subheader("UPI Transaction Patterns")
        upi_transaction_count = st.number_input("Monthly UPI Transactions Count", 5, 200, 30)
        avg_upi_amount = st.number_input("Average UPI Transaction Amount (₹)", 
                                       min_value=100, 
                                       max_value=15000, 
                                       value=1000)
        monthly_upi_spend = st.number_input("Monthly UPI Spend (₹)", 
                                          min_value=500, 
                                          max_value=100000, 
                                          value=30000)
        merchant_diversity = st.slider("Number of Different UPI Merchants", 1, 10, 5)
        
        # E-commerce Behavior
        st.subheader("E-commerce Activity")
        monthly_ecom_spend = st.number_input("Monthly E-commerce Spend (₹)", 
                                           min_value=1000, 
                                           max_value=80000, 
                                           value=10000)
        avg_cart_value = st.number_input("Average Cart Value (₹)", 
                                       min_value=500, 
                                       max_value=25000, 
                                       value=2000)
        num_orders = st.number_input("Number of Monthly Orders", 1, 40, 5)
        category_diversity = st.slider("Number of Different Shopping Categories", 1, 8, 4)
        payment_preference = st.selectbox("Preferred Payment Method",
                                        ['UPI', 'Credit Card', 'Debit Card', 'COD', 'EMI'])
        
        # Investment Portfolio
        st.subheader("Investment Profile")
        total_investment = st.number_input("Total Investment Amount (₹)", 
                                         min_value=0, 
                                         max_value=2000000, 
                                         value=100000)
        num_stocks = st.number_input("Number of Stocks/Mutual Funds", 0, 50, 5)
        portfolio_risk = st.slider("Portfolio Risk Level", 0.0, 1.0, 0.5,
                                 help="0 = Low Risk, 1 = High Risk")
        avg_holding_period = st.slider("Average Investment Holding Period (months)", 1, 60, 12)
        profit_loss_ratio = st.slider("Profit/Loss Ratio", 0.0, 2.0, 1.0,
                                    help="Values above 1 indicate more profits than losses")
        sip_active = st.selectbox("Active SIP Investments", ["Yes", "No"])
        investment_maturity_score = st.slider("Investment Maturity Score", 1, 10, 5,
                                           help="1 = Novice, 10 = Expert")
        
        # Financial Health
        st.subheader("Financial Health Indicators")
        late_payment_count = st.number_input("Number of Late Payments (Last 12 months)", 0, 12, 0)
        credit_utilization = st.slider("Credit Utilization (%)", 0.0, 100.0, 30.0)
        savings_balance = st.number_input("Savings Balance (₹)", 
                                        min_value=0, 
                                        max_value=10000000, 
                                        value=50000)
        monthly_savings_rate = st.slider("Monthly Savings Rate (%)", 0.0, 100.0, 20.0)
        num_bank_accounts = st.number_input("Number of Bank Accounts", 1, 10, 1)
        insurance_policies = st.number_input("Number of Insurance Policies", 0, 10, 1)
        
        # Financial Ratios
        st.subheader("Financial Ratios")
        debt_to_income_ratio = st.slider("Debt to Income Ratio (%)", 0.0, 100.0, 30.0)
        income_to_spend_ratio = st.slider("Income to Spending Ratio", 0.1, 5.0, 2.0,
                                        help="Higher values indicate better saving habits")
        savings_stability_index = st.slider("Savings Stability Index", 1, 10, 5,
                                          help="1 = Highly Variable, 10 = Very Stable")
        financial_health_score = st.slider("Overall Financial Health Score", 1, 100, 70,
                                         help="1 = Poor, 100 = Excellent")
        
        # Submit button with save option
        col1, col2 = st.columns(2)
        with col1:
            save_assessment = st.checkbox("Save assessment results", value=False)
        
        submitted = st.form_submit_button("Analyze Credit Risk")
        
        if submitted:
            # Prepare input data
            # Map user input to match the model's expected features
            user_input = {
                'age': age,
                'occupation': occupation,
                'income': annual_income,
                'city': city,
                'education': education,
                'employment_length': employment_length,
                'existing_loans': existing_loans,
                'credit_history_length': account_age,
                'upi_transaction_count': upi_transaction_count,
                'avg_upi_amount': avg_upi_amount,
                'monthly_upi_spend': monthly_upi_spend,
                'transaction_volatility': 0.5,  # Default value
                'income_ratio': monthly_upi_spend / annual_income,
                'merchant_diversity': merchant_diversity,
                'monthly_ecom_spend': monthly_ecom_spend,
                'avg_cart_value': avg_cart_value,
                'num_orders': num_orders,
                'return_rate': 0.1,  # Default value
                'category_diversity': category_diversity,
                'payment_preference': payment_preference,
                'num_stocks': num_stocks,
                'total_investment': total_investment,
                'portfolio_risk': portfolio_risk,
                'avg_holding_period': avg_holding_period,
                'profit_loss_ratio': profit_loss_ratio,
                'sip_active': 1 if sip_active == "Yes" else 0,
                'late_payment_count': late_payment_count,
                'credit_utilization': credit_utilization,
                'savings_balance': savings_balance,
                'monthly_savings_rate': monthly_savings_rate,
                'num_bank_accounts': num_bank_accounts,
                'insurance_policies': insurance_policies,
                'debt_to_income_ratio': debt_to_income_ratio,
                'income_to_spend_ratio': income_to_spend_ratio,
                'savings_stability_index': savings_stability_index,
                'investment_maturity_score': investment_maturity_score,
                'financial_health_score': financial_health_score
            }
            
            # Preprocess and predict
            input_processed = preprocess_input(user_input)
            prediction_proba = model.predict_proba(input_processed)[0]
            prediction = model.predict(input_processed)[0]
            
            # Display results
            st.subheader("Risk Assessment Results")
            
            # Create risk score (0-100)
            risk_score = int((1 - prediction_proba[1]) * 100)
            
            # Display risk score with color coding
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{risk_score}/100")
            
            with col2:
                risk_level = "Low Risk" if prediction == 0 else "High Risk"
                risk_color = "green" if prediction == 0 else "red"
                st.markdown(f"<h3 style='color: {risk_color};'>{risk_level}</h3>", 
                          unsafe_allow_html=True)
            
            with col3:
                confidence = max(prediction_proba) * 100
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Additional risk insights
            st.subheader("Risk Insights")
            
            # Financial Health Indicators
            if financial_health_score < 40:
                st.error("🚨 Poor overall financial health")
            elif financial_health_score > 80:
                st.success("✅ Excellent financial health")
            
            # Credit Utilization
            if credit_utilization > 70:
                st.error("🚨 Extremely high credit utilization")
            elif credit_utilization > 50:
                st.warning("⚠️ High credit utilization")
            elif credit_utilization < 30:
                st.success("✅ Healthy credit utilization")
            
            # Debt to Income Ratio
            if debt_to_income_ratio > 50:
                st.error("🚨 High debt burden")
            elif debt_to_income_ratio < 30:
                st.success("✅ Healthy debt levels")
            
            # Savings and Investment Behavior
            if monthly_savings_rate < 10:
                st.warning("⚠️ Low savings rate")
            elif monthly_savings_rate > 30:
                st.success("✅ Strong saving habits")
            
            if sip_active == "Yes":
                st.success("✅ Regular investment through SIP")
            
            if profit_loss_ratio > 1.5:
                st.success("✅ Strong investment performance")
            elif profit_loss_ratio < 0.5:
                st.warning("⚠️ Poor investment returns")
            
            # Transaction and Payment History
            if late_payment_count > 2:
                st.error("🚨 Multiple late payments indicate risk")
            elif late_payment_count == 0:
                st.success("✅ Perfect payment history")
            
            # Spending Patterns
            monthly_income = annual_income / 12
            if monthly_upi_spend > monthly_income * 0.7:
                st.warning("⚠️ High UPI spending relative to income")
            
            if monthly_ecom_spend > monthly_income * 0.4:
                st.warning("⚠️ High e-commerce spending relative to income")
            
            # Investment Maturity
            if investment_maturity_score > 7:
                st.success("✅ Mature investment approach")
            elif investment_maturity_score < 3:
                st.info("ℹ️ Novice investor - consider financial education")
            
            # Savings Stability
            if savings_stability_index < 3:
                st.warning("⚠️ Unstable savings pattern")
            elif savings_stability_index > 7:
                st.success("✅ Very stable savings pattern")
            
            # Save assessment if requested
            if save_assessment:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    assessment_data = {
                        "timestamp": timestamp,
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "confidence": confidence,
                        **user_input
                    }
                    
                    # Create assessments directory if it doesn't exist
                    if not os.path.exists("assessments"):
                        os.makedirs("assessments")
                    
                    # Save assessment with timestamp
                    filename = f"assessments/assessment_{timestamp}.csv"
                    pd.DataFrame([assessment_data]).to_csv(filename, index=False)
                    st.success(f"✅ Assessment saved successfully as {filename}")
                except Exception as e:
                    st.error(f"❌ Error saving assessment: {str(e)}")

if __name__ == "__main__":
    main()