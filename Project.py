import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import io


st.set_page_config(
    page_title="Data Distribution Analyzer", 
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');
    
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        background: #0299ba;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -1px;
    }
    
    
    .section-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: #764ba2;
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button {
        background: #00b4db;
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 25px;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 180, 219, 0.4);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
    }
    
    .distribution-checkbox {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 4px solid #00b4db;
    }
</style>
""", unsafe_allow_html=True)


if 'data_array' not in st.session_state:
    st.session_state.data_array = None
if 'fitted_params' not in st.session_state:
    st.session_state.fitted_params = []
if 'distribution_names' not in st.session_state:
    st.session_state.distribution_names = []
if 'selected_distributions' not in st.session_state:
    st.session_state.selected_distributions = []
if 'errors' not in st.session_state:
    st.session_state.errors = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Data Input"


st.markdown('<h1 class="main-title">DATA DISTRIBUTION ANALYZER</h1>', unsafe_allow_html=True)



nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    if st.button("DATA INPUT", use_container_width=True):
        st.session_state.current_tab = "Data Input"
with nav_col2:
    if st.button("VISUALIZE", use_container_width=True):
        st.session_state.current_tab = "Visualize"
with nav_col3:
    if st.button("FIT MODELS", use_container_width=True):
        st.session_state.current_tab = "Fit Models"
with nav_col4:
    if st.button("RESULTS", use_container_width=True):
        st.session_state.current_tab = "Results"

st.markdown("---")


if st.session_state.current_tab == "Data Input":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Data Source Selection")
        data_source = st.radio("Choose your data source:", 
                              ["Manual Entry", "Upload CSV"],
                              horizontal=True)
        
        if data_source == "Manual Entry":
            st.markdown("**Enter your data values:**")
            manual_data = st.text_area(
                "Separate values with commas:",
                value="2.1, 3.8, 4.2, 5.7, 6.3, 7.9, 8.4, 9.6, 10.2, 11.8, 3.1, 4.5, 5.2, 6.8, 7.1, 8.9, 9.3, 10.7, 11.4, 12.1",
                height=120
            )
            if st.button("Process Manual Data", key="process_manual"):
                try:
                    data_list = [float(x.strip()) for x in manual_data.split(',')]
                    st.session_state.data_array = np.array(data_list)
                    st.success(f"✅ Processed {len(data_list)} data points")
                except ValueError:
                    st.error("❌ Please enter valid numbers separated by commas")
        
        elif data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your CSV file:", type="csv")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("**File Preview:**")
                    st.dataframe(df.head(3))
                    
                    if len(df.columns) > 1:
                        selected_col = st.selectbox("Select data column:", df.columns)
                        st.session_state.data_array = df[selected_col].values
                    else:
                        st.session_state.data_array = df.iloc[:, 0].values
                    
                    st.success(f"✅ Loaded {len(st.session_state.data_array)} rows")
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
        

    
    with col2:
        st.markdown("### Data Overview")
        if st.session_state.data_array is not None:

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Points</h3>
                    <h2>{len(st.session_state.data_array)}</h2>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card" style="background: #00f2fe;">
                    <h3>Mean</h3>
                    <h2>{np.mean(st.session_state.data_array):.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card" style="background: #36e630;">
                    <h3>Std Dev</h3>
                    <h2>{np.std(st.session_state.data_array):.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card" style="background: #fa709a;">
                    <h3>Range</h3>
                    <h2>{np.min(st.session_state.data_array):.1f}-{np.max(st.session_state.data_array):.1f}</h2>
                </div>
                """, unsafe_allow_html=True)
            

            st.markdown("**Initial Distribution Insight:**")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(st.session_state.data_array, bins=20, alpha=0.7, color='#00b4db', edgecolor='white')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Select a data source and load data to see overview")
    
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.current_tab == "Visualize":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    if st.session_state.data_array is None:
        st.warning("Please load data in the Data Input section first")
    else:
        st.markdown("### Data Scatter Plot")
        
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        
        measurement_numbers = np.arange(1, len(st.session_state.data_array) + 1)
        scatter = ax.scatter(measurement_numbers, st.session_state.data_array, 
                           alpha=0.6, color='#00b4db', s=30, label='Data Points')
        
        ax.set_xlabel('Measurement Number', fontsize=12, fontweight='medium')
        ax.set_ylabel('Value', fontsize=12, fontweight='medium')
        ax.set_title('Raw Data Scatter Plot', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(st.session_state.data_array) * 1.1)
        

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        st.pyplot(fig)
    
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.current_tab == "Fit Models":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    if st.session_state.data_array is None:
        st.warning("Please load data first to fit distributions")
    else:
        st.markdown("### Select Distributions to Fit")
        

        col1, col2, col3 = st.columns(3)
        

        with col1:
            st.markdown("**Common Distributions**")
            norm_dist = st.checkbox("Normal Distribution", value=True, key="norm")
            gamma_dist = st.checkbox("Gamma Distribution", value=True, key="gamma")
            weibull_dist = st.checkbox("Weibull Distribution", value=True, key="weibull")
            expon_dist = st.checkbox("Exponential Distribution", key="expon")
        

        with col2:
            st.markdown("**Intermediate Distributions**")
            lognorm_dist = st.checkbox("Log-Normal Distribution", key="lognorm")
            beta_dist = st.checkbox("Beta Distribution", key="beta")
            laplace_dist = st.checkbox("Laplace Distribution", key="laplace")
            gumbel_dist = st.checkbox("Gumbel Distribution", key="gumbel")
        

        with col3:
            st.markdown("**Advanced Distributions**")  
            cauchy_dist = st.checkbox("Cauchy Distribution", key="cauchy")
            rayleigh_dist = st.checkbox("Rayleigh Distribution", key="rayleigh")
            uniform_dist = st.checkbox("Uniform Distribution", key="uniform")
        

        selected_distributions = []
        distribution_names = []
        
        if norm_dist:
            selected_distributions.append(stats.norm)
            distribution_names.append("Normal")
        if gamma_dist:
            selected_distributions.append(stats.gamma)
            distribution_names.append("Gamma")
        if weibull_dist:
            selected_distributions.append(stats.weibull_min)
            distribution_names.append("Weibull")
        if expon_dist:
            selected_distributions.append(stats.expon)
            distribution_names.append("Exponential")
        if lognorm_dist:
            selected_distributions.append(stats.lognorm)
            distribution_names.append("Log-Normal")
        if beta_dist:
            selected_distributions.append(stats.beta)
            distribution_names.append("Beta")
        if laplace_dist:
            selected_distributions.append(stats.laplace)
            distribution_names.append("Laplace")
        if gumbel_dist:
            selected_distributions.append(stats.gumbel_r)
            distribution_names.append("Gumbel")
        if cauchy_dist:
            selected_distributions.append(stats.cauchy)
            distribution_names.append("Cauchy")
        if rayleigh_dist:
            selected_distributions.append(stats.rayleigh)
            distribution_names.append("Rayleigh")
        if uniform_dist:
            selected_distributions.append(stats.uniform)
            distribution_names.append("Uniform")
        

        if st.button("Fit Selected Distributions", use_container_width=True):
            if selected_distributions:
                with st.spinner("Fitting distributions..."):
                    fitted_params = []
                    errors = []
                    
                    hist, bin_edges = np.histogram(st.session_state.data_array, bins=30, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    progress_bar = st.progress(0)
                    for i, (dist, name) in enumerate(zip(selected_distributions, distribution_names)):
                        try:
                            params = dist.fit(st.session_state.data_array)
                            fitted_params.append(params)
                            fitted_dist = dist(*params)
                            pdf_values = fitted_dist.pdf(bin_centers)
                            error = np.mean(np.abs(hist - pdf_values))
                            errors.append(error)
                        except:
                            fitted_params.append(None)
                            errors.append(float('inf'))
                        
                        progress_bar.progress((i + 1) / len(selected_distributions))
                    
                    st.session_state.fitted_params = fitted_params
                    st.session_state.distribution_names = distribution_names
                    st.session_state.selected_distributions = selected_distributions
                    st.session_state.errors = errors
                    st.session_state.bin_centers = bin_centers
                    st.session_state.hist = hist
                
                st.success(f"✅ Successfully fitted {len(selected_distributions)} distributions!")
            else:
                st.error("❌ Please select at least one distribution")
    
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.current_tab == "Results":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    if not st.session_state.fitted_params:
        st.warning("Please fit distributions in the Fit Models section first")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Fitting Results")
            

            results_data = []
            for i, (name, error, params) in enumerate(zip(
                st.session_state.distribution_names,
                st.session_state.errors,
                st.session_state.fitted_params
            )):
                if params is not None:
                    results_data.append({
                        'Distribution': name,
                        'Avg Error': f"{error:.4f}",
                        'Params': len(params)
                    })
            
            results_df = pd.DataFrame(results_data)
            

            if len(results_df) > 0:
                best_idx = np.argmin(st.session_state.errors)
                best_name = st.session_state.distribution_names[best_idx]
                best_error = st.session_state.errors[best_idx]
                
                st.markdown(f"""
                <div style='background: #00b4db; 
                          padding: 1.5rem; border-radius: 12px; color: white; margin: 1rem 0;'>
                    <h3>Best Fit</h3>
                    <h2>{best_name}</h2>
                    <p>Average Error: {best_error:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(results_df.style.highlight_min(subset=['Avg Error']), use_container_width=True)
        
        with col2:
            st.markdown("### Comparison Visualization")
            

            fig, ax = plt.subplots(figsize=(10, 6))
            

            ax.hist(st.session_state.data_array, bins=30, density=True, 
                   alpha=0.3, color='gray', label='Data', edgecolor='black')
            

            x_plot = np.linspace(np.min(st.session_state.data_array), 
                               np.max(st.session_state.data_array), 1000)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(st.session_state.selected_distributions)))
            
            for i, (dist, name, params) in enumerate(zip(
                st.session_state.selected_distributions,
                st.session_state.distribution_names,
                st.session_state.fitted_params
            )):
                if params is not None:
                    try:
                        fitted_dist = dist(*params)
                        pdf_values = fitted_dist.pdf(x_plot)
                        ax.plot(x_plot, pdf_values, linewidth=2.5, 
                               color=colors[i], label=f'{name} (err: {st.session_state.errors[i]:.3f})')
                    except:
                        continue
            
            ax.set_xlabel('Value', fontweight='bold')
            ax.set_ylabel('Probability Density', fontweight='bold')
            ax.set_title('Distribution Fitting Comparison', fontweight='bold', pad=20)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            

            st.markdown("### Parameter Tuning")
            selected_dist = st.selectbox("Select distribution to tune:", st.session_state.distribution_names)
            
            if selected_dist:
                idx = st.session_state.distribution_names.index(selected_dist)
                dist_obj = st.session_state.selected_distributions[idx]
                params = st.session_state.fitted_params[idx]
                
                tuned_params = []
                for i, param in enumerate(params):
                    tuned_param = st.slider(f"Parameter {i+1}", 
                                          value=float(param),
                                          min_value=float(param * 0.1),
                                          max_value=float(param * 2.0),
                                          key=f"tune_{selected_dist}_{i}")
                    tuned_params.append(tuned_param)
                
                if st.button("Update Fit"):

                    tuned_dist = dist_obj(*tuned_params)
                    pdf_tuned = tuned_dist.pdf(st.session_state.bin_centers)
                    new_error = np.mean(np.abs(st.session_state.hist - pdf_tuned))
                    old_error = st.session_state.errors[idx]
                    
                    if new_error < old_error:
                        st.success(f"Improved! Error: {old_error:.4f} → {new_error:.4f}")
                    else:
                        st.info(f"Current error: {new_error:.4f} (original: {old_error:.4f})")
    
    st.markdown('</div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("""
    <div style='background: #00b4db; 
              padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;'>
        <h2>NE 111 Webapp Project</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Quick Actions")
    if st.button("Reset All Data", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    if st.button("Generate Random Sample", use_container_width=True):
        st.session_state.data_array = stats.gamma.rvs(5, 1, 1, size=1000)
        st.session_state.current_tab = "Visualize"
        st.rerun()
    
    st.markdown("### Features Included:")
    st.markdown("""
    - **A way for users to enter data**
        - **A data entry area for users to enter data by hand**
        - **An option to upload data in the form of a CSV file**
    - **The ability to fit multiple different types of distributions** 
        - **Has 10 different options**
    - **A visualization of the data and the fitted distribution**
    - **An output area that shows the fitting parameters and information about the quality of the fit**
    - **Manual fitting option**
    - **Professional analytics**
    """)
    
    


