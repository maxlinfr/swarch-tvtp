import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

random_guess = True #can set to false if time-constrained
random_guess_number = 3

def extract_prepare_data():

    #1. Fetch raw data from Yahoo Finance
    print("Fetching data from Yahoo Finance...")
    tickers = ["^GSPC","^VIX"]
    data = yf.download(tickers, start="2010-01-01", end="2022-12-31")['Close']

    #2 Calculate log returns for the s&p
    print("Calculating log returns...")
    df = pd.DataFrame()
    df['SPY_Returns'] = np.log(data['^GSPC'] / data['^GSPC'].shift(1))*100

    #3 Using VIX
    df['VIX_Lagged'] = data['^VIX'].shift(1)

    #4 Drop NaN values
    df.dropna(inplace=True)
    return df

#Hamilton Filter Part
def run_hamilton_filter(params, returns, vix):
    T = len(returns)
    omega, alpha, beta, lambda_g, gamma, b0_01, b1_01, b0_10, b1_10 = params

    # --- Precompute GARCH variance (can't vectorize, each step needs previous) ---
    # Precompute indicator array
    indicator = (returns < 0).astype(float)

    variance = np.zeros(T)
    variance[0] = omega / (1 - alpha - beta - 0.5 * lambda_g)
    
    for t in range(1, T):
        variance[t] = (omega 
                    + alpha   * (returns[t-1]**2) 
                    + lambda_g * indicator[t-1] * (returns[t-1]**2) 
                    + beta    * variance[t-1])
    var_state0 = variance
    var_state1 = variance * gamma

    # --- Precompute everything that doesn't depend on t ---
    p01 = 1.0 / (1.0 + np.exp(-(b0_01 + b1_01 * vix)))
    p10 = 1.0 / (1.0 + np.exp(-(b0_10 + b1_10 * vix)))
    p00 = 1.0 - p01
    p11 = 1.0 - p10

    std_state0 = np.sqrt(np.maximum(var_state0, 1e-6))
    std_state1 = np.sqrt(np.maximum(var_state1, 1e-6))

    # --- Allocate output arrays ---
    filtered_probs  = np.zeros((T, 2))
    predicted_probs = np.zeros((T, 2))
    likelihoods     = np.zeros(T)

    #hard-coded likelihood but over large periods of data it will not matter much, as the filter will converge to the correct state probabilities.
    predicted_probs[0, 0] = 0.90
    predicted_probs[0, 1] = 0.10

    # --- Loop now only does the Bayesian update (unavoidably sequential) ---
    for t in range(T):
        lik_state0 = max(norm.pdf(returns[t], 0.0, std_state0[t]), 1e-10)
        lik_state1 = max(norm.pdf(returns[t], 0.0, std_state1[t]), 1e-10)

        blended_lik = predicted_probs[t, 0] * lik_state0 + predicted_probs[t, 1] * lik_state1
        likelihoods[t] = blended_lik

        filtered_probs[t, 0] = (predicted_probs[t, 0] * lik_state0) / blended_lik
        filtered_probs[t, 1] = (predicted_probs[t, 1] * lik_state1) / blended_lik

        if t < T - 1:
            predicted_probs[t+1, 0] = filtered_probs[t, 0] * p00[t] + filtered_probs[t, 1] * p10[t]
            predicted_probs[t+1, 1] = filtered_probs[t, 0] * p01[t] + filtered_probs[t, 1] * p11[t]

            predicted_probs[t+1] = np.clip(predicted_probs[t+1], 1e-10, 1.0 - 1e-10)
            predicted_probs[t+1] /= np.sum(predicted_probs[t+1])

    return likelihoods, filtered_probs

def objective_function(params, returns, vix):
    """
    The function scipy.optimize tries to minimize.
    Because we want to MAXIMIZE the Log-Likelihood, we return the NEGATIVE Log-Likelihood.
    """
    # Soft constraint check: GARCH parameters must be positive, gamma must be > 1.0
    omega, alpha, beta, lambda_g, gamma = params[0], params[1], params[2], params[3], params[4]
    if (omega <= 0 or alpha <= 0 or beta <= 0 or lambda_g <= 0 or
    (alpha + lambda_g/2 + beta) >= 1.0 or gamma <= 1.0):
        return 1e10
    likelihoods, _ = run_hamilton_filter(params, returns, vix)
    
    # Calculate Sum of Log-Likelihoods
    log_likelihood = np.sum(np.log(likelihoods))
    
    # Return negative value so the minimizer maximizes the true likelihood
    return -log_likelihood

if __name__ == "__main__":
    # Prepare Data
    df = extract_prepare_data()
    returns = df['SPY_Returns'].values
    vix = df['VIX_Lagged'].values
    
    # Initial Parameter Guesses [omega, alpha, gamma, b0_01, b1_01, b0_10, b1_10]
    # We guide the optimizer with standard, historically reasonable values
    
    # Define bounds to prevent the math from exploding into impossible dimensions
    bounds = [
        (1e-4, 2.0),   # omega
        (1e-4, 0.99),  # alpha
        (1e-4, 0.99),  # beta
        (1e-4, 0.5),  # lambda_g
        (1.1, 10.0),   # gamma
        (-10.0, 10.0), # b0_01
        (-1.0, 1.0),   # b1_01
        (-10.0, 10.0), # b0_10
        (-1.0, 1.0)    # b1_10
    ]
    
    best_result = None
    best_ll = np.inf

    print("\nStep 2: Running random restarts for Maximum Likelihood Estimation...")
    n_years = (pd.Timestamp("2022-12-31") - pd.Timestamp("2010-01-01")).days / 365.25
    print(f"This runs {random_guess_number} restarts over {n_years:.0f} years of market data...")

    if random_guess:
        for i in range(random_guess_number):
            while True:
                a  = np.random.uniform(1e-4, 0.15)
                lg = np.random.uniform(1e-4, 0.15)
                b  = np.random.uniform(0.7,  0.97)
                if (a + lg/2 + b) < 1.0:
                    break
            guess = [
                np.random.uniform(1e-4, 0.5),    # omega
                a,
                b,
                lg,    # lambda_g
                np.random.uniform(1.5, 6.0),     # gamma (tightened upper bound)
                np.random.uniform(-5.0, 0.0),    # b0_01
                np.random.uniform(0.0, 0.3),     # b1_01
                np.random.uniform(-5.0, 0.0),    # b0_10
                np.random.uniform(-0.2, 0.0),    # b1_10
            ]

            result = minimize(
                objective_function,
                guess,
                args=(returns, vix),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500}
            )
            
            print(f"  Restart {i+1}/{random_guess_number} — Log-Likelihood: {-result.fun:.2f} | Converged: {result.success}")
            
            if result.fun < best_ll:
                best_ll = result.fun
                best_result = result
    else:
        # Use a single, reasonable initial guess if random_guess is False
        initial_guess = [
            0.15,  # omega
            0.06,  # alpha  ← nudge down slightly since lambda_g carries some of the load
            0.85,  # beta
            0.08,  # lambda_g  ← asymmetry term, typically 0.05-0.15 for equities
            3.5,   # gamma
            -2.5,  # b0_01
            0.10,  # b1_01
            -2.0,  # b0_10
            -0.05  # b1_10
        ]
        best_result = minimize(
            objective_function,
            initial_guess,
            args=(returns, vix),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500}
        )
        best_ll = best_result.fun

    print(f"\nBest Log-Likelihood found: {-best_ll:.2f}")
    opt_result = best_result  # rest of your code stays identical
    
    print(opt_result.success)   # should be True
    print(opt_result.message)   # should say converged
    
    # Unpack optimal parameters (8-parameter version)
    opt_params = opt_result.x
    print("\n=== OPTIMAL MODEL PARAMETERS ===")
    print(f"Optimal GARCH-GJR Omega: {opt_params[0]:.4f}")
    print(f"Optimal GARCH-GJR Alpha: {opt_params[1]:.4f}")
    print(f"Optimal GARCH-GJR Beta:  {opt_params[2]:.4f}") 
    print(f"Optimal Lambda_g: {opt_params[3]:.4f}")
    print(f"Optimal State 1 Vol Multiplier (Gamma): {opt_params[4]:.4f}x")  
    print(f"Transition (Bull -> Bear) VIX Sensitivity: {opt_params[6]:.4f}") 
    print(f"Transition (Bear -> Bull) VIX Sensitivity: {opt_params[8]:.4f}")
    # Run Hamilton Filter one final time using our optimal parameters
    _, final_filtered_probs = run_hamilton_filter(opt_params, returns, vix)
    
    # Add our model's beliefs back to our dataframe
    df['Prob_Bear_State'] = final_filtered_probs[:, 1]
    
    #opt_params for backtesting notebook
    print(repr(opt_params.tolist()))
