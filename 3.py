import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf

st.set_page_config(page_title="Slurry 조성 최적화 GP", layout="wide")
st.title("Slurry 조성 최적화 GP")

CSV_PATH = "slurry_data_wt%_ALL.csv"
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    df = pd.DataFrame(columns=["carbon_black_wt%", "graphite_wt%", "CMC_wt%", "solvent_wt%", "yield_stress"])

# --- 사용자 입력 폼 (Graphite 자동 계산) ---
st.sidebar.header("새로운 실험 조성 추가")
with st.sidebar.form("new_data_form"):
    new_cb = st.number_input("Carbon Black [wt%]", min_value=0.0, step=0.1)
    new_cmc = st.number_input("CMC [wt%]", min_value=0.0, step=0.05)
    new_solvent = st.number_input("Solvent [wt%]", min_value=0.0, step=0.5)

    total_input = new_cb + new_cmc + new_solvent
    new_graphite = max(0.0, 100.0 - total_input)  # 음수 방지
    st.markdown(f"Graphite: **{new_graphite:.2f} wt%**")

    new_yield = st.number_input("Yield Stress [Pa]", min_value=0.0, step=10.0)
    submitted = st.form_submit_button("데이터 추가")

if submitted:
    if total_input > 100:
        st.sidebar.error("⚠️ Carbon Black + CMC + Solvent의 합이 100을 초과했습니다.")
    else:
        new_row = {
            "carbon_black_wt%": new_cb,
            "graphite_wt%": new_graphite,
            "CMC_wt%": new_cmc,
            "solvent_wt%": new_solvent,
            "yield_stress": new_yield,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        st.sidebar.success("✅ 데이터가 저장되었습니다.")

# --- 데이터 전처리 ---
x_cols = ["carbon_black_wt%", "graphite_wt%", "CMC_wt%", "solvent_wt%"]
y_cols = ["yield_stress"]
X_raw = df[x_cols].values
Y_raw = df[y_cols].values

param_bounds = {
    "carbon_black_wt%": (1.75, 10.0),
    "graphite_wt%": (18.0, 38.0),
    "CMC_wt%": (0.7, 1.5),
    "solvent_wt%": (58.0, 78.0),
}
bounds_array = np.array([param_bounds[k] for k in x_cols])
x_scaler = MinMaxScaler()
x_scaler.fit(bounds_array.T)

X_scaled = x_scaler.transform(X_raw)
train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double)

# --- 모델 학습 ---
model = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# --- 제약조건 설정 ---
input_dim = train_x.shape[1]
bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim], dtype=torch.double)
scales = bounds_array[:, 1] - bounds_array[:, 0]
offset = np.sum(bounds_array[:, 0])
rhs = 100.0 - offset
indices = torch.arange(len(x_cols), dtype=torch.long)
coefficients = torch.tensor(scales, dtype=torch.double)
rhs_tensor = torch.tensor(rhs, dtype=torch.double)
inequality_constraints = [
    (indices, coefficients, rhs_tensor),
    (indices, -coefficients, -rhs_tensor),
]


# --- 중복 확인 ---
def is_duplicate(candidate_scaled, train_scaled, tol=1e-3):
    return any(np.allclose(candidate_scaled, x, atol=tol) for x in train_scaled)


# --- 추천 실행 ---
candidate_wt = None
if st.button("Candidate"):
    best_y = train_y.max().item()
    acq_fn = ExpectedImprovement(model=model, best_f=best_y, maximize=True)

    for _ in range(10):
        candidate_scaled, _ = optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
            inequality_constraints=inequality_constraints,
        )
        candidate_np = candidate_scaled.detach().numpy()[0]
        if is_duplicate(candidate_np, train_x.numpy()):
            continue
        x_tensor = torch.tensor(candidate_np.reshape(1, -1), dtype=torch.double)
        y_pred = model.posterior(x_tensor).mean.item()

        if y_pred > 0:
            candidate_wt = x_scaler.inverse_transform(candidate_np.reshape(1, -1))[0]
            break

    if candidate_wt is not None:
        st.subheader("Candidate")
        for i, col in enumerate(x_cols):
            st.write(f"{col}: **{candidate_wt[i]:.2f} wt%**")
        st.write(f"**총합**: {np.sum(candidate_wt):.2f} wt%")
        st.write(f"**예측 Yield Stress**: {y_pred:.2f} Pa")
    else:
        st.warning("Yield Stress > 0 조건을 만족하는 조성을 찾지 못했습니다.")

# --- 예측 곡선 시각화 ---
cb_idx = x_cols.index("carbon_black_wt%")
x_vals_scaled = np.linspace(0, 1, 100)
mean_scaled = np.mean(X_scaled, axis=0)
X_test_scaled = np.tile(mean_scaled, (100, 1))
X_test_scaled[:, cb_idx] = x_vals_scaled
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.double)

model.eval()
with torch.no_grad():
    posterior = model.posterior(X_test_tensor)
    mean = posterior.mean.numpy().flatten()
    std = posterior.variance.sqrt().numpy().flatten()

cb_vals_wt = x_scaler.inverse_transform(X_test_scaled)[:, cb_idx]
train_x_cb = x_scaler.inverse_transform(train_x.numpy())[:, cb_idx]
train_y_np = train_y.numpy().flatten()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cb_vals_wt, mean, label="Predicted Mean", color="blue")
ax.fill_between(cb_vals_wt, mean - 1.96 * std, mean + 1.96 * std, color="blue", alpha=0.2, label="95% CI")
ax.scatter(train_x_cb, train_y_np, color="red", label="Observed Data")
if candidate_wt is not None:
    cand_scaled = x_scaler.transform(candidate_wt.reshape(1, -1))
    pred_y = model.posterior(torch.tensor(cand_scaled, dtype=torch.double)).mean.item()
    ax.scatter(candidate_wt[cb_idx], pred_y, color="yellow", label="Candidate")

ax.set_xlabel("Carbon Black [wt%]")
ax.set_ylabel("Yield Stress [Pa]")
ax.set_title("GP Prediction")
ax.grid(True)
ax.legend()
st.pyplot(fig)
