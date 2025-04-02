import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf

st.set_page_config(page_title="Carbon Black Optimization", layout="wide")
st.title("Slurry 조성 최적화 GP")

# 1. 데이터 불러오기
df = pd.read_csv("slurry_data_wt%.csv")

# 2. 입력/출력 설정
x_cols = ["carbon_black_wt%", "graphite_wt%", "CMC_wt%", "solvent_wt%"]
y_cols = ["yield_stress"]
X_raw = df[x_cols].values
Y_raw = df[y_cols].values

# 3. 실험 DOE 기반 + 완만히 확장된 추천 범위
param_bounds = {
    "carbon_black_wt%": (0.5, 5.0),
    "CMC_wt%":          (0.7, 1.5),
    "solvent_wt%":      (58.0, 78.0),
    "graphite_wt%":     (18.0, 38.0),
}

# 4. 수동 정규화 함수
def normalize(X, bounds_array):
    return (X - bounds_array[:, 0]) / (bounds_array[:, 1] - bounds_array[:, 0])

def denormalize(X_scaled, bounds_array):
    return X_scaled * (bounds_array[:, 1] - bounds_array[:, 0]) + bounds_array[:, 0]

# 5. bounds array 생성
bounds_array = np.array([param_bounds[key] for key in x_cols])
X_scaled = normalize(X_raw, bounds_array)
train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw, dtype=torch.double)

# 6. GP 모델 학습
model = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# 7. 탐색 범위 및 제약조건 설정
input_dim = train_x.shape[1]
bounds = torch.stack([
    torch.zeros(input_dim, dtype=torch.double),
    torch.ones(input_dim, dtype=torch.double)
])

# ✅ equality 제약조건: 정규화된 조성들의 실제 합이 100이 되도록
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

candidate_wt = None

# 8. 버튼을 눌렀을 때 추천 수행
if st.button("Candidate"):
    best_y = train_y.max()
    acq_fn = LogExpectedImprovement(model=model, best_f=best_y, maximize=True)

    y_pred = -float("inf")
    for attempt in range(5):
        candidate_scaled, _ = optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
            inequality_constraints=inequality_constraints,
        )
        candidate_np = candidate_scaled.detach().numpy()
        candidate_temp = denormalize(candidate_np, bounds_array)[0]

        x_norm = normalize(candidate_temp.reshape(1, -1), bounds_array)
        x_tensor = torch.tensor(x_norm, dtype=torch.double)
        y_pred = model.posterior(x_tensor).mean.item()

        if y_pred > 0:
            candidate_wt = candidate_temp
            break

    if candidate_wt is not None:
        st.subheader("Candidate")
        for i, col in enumerate(x_cols):
            st.write(f"{col}: **{candidate_wt[i]:.2f} wt%**")
        st.write(f"**총합**: {np.sum(candidate_wt):.2f} wt%")
        st.write(f"**예측 Yield Stress**: {y_pred:.2f} Pa")
    else:
        st.warning("Yield Stress > 0 조건을 만족하는 추천 조성을 찾지 못했습니다.")

# 9. Carbon Black 변화에 따른 예측 곡선
dx = x_cols.index("carbon_black_wt%")
x_vals = np.linspace(0, 1, 100)
mean_scaled = np.mean(X_scaled, axis=0)
X_test_scaled = np.tile(mean_scaled, (100, 1))
X_test_scaled[:, dx] = x_vals
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.double)

model.eval()
with torch.no_grad():
    posterior = model.posterior(X_test_tensor)
    mean = posterior.mean.numpy().flatten()
    std = posterior.variance.sqrt().numpy().flatten()

carbon_black_vals_wt = denormalize(X_test_scaled, bounds_array)[:, dx]
train_x_wt = denormalize(train_x.numpy(), bounds_array)[:, dx]
train_y_np = train_y.numpy().flatten()

# 10. 그래프 출력
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(carbon_black_vals_wt, mean, label="Predicted Mean", color="blue")
ax.fill_between(carbon_black_vals_wt, mean - 1.96 * std, mean + 1.96 * std, color="blue", alpha=0.2, label="95% CI")
ax.scatter(train_x_wt, train_y_np, color="red", label="Observed Data")
if candidate_wt is not None:
    ax.scatter(
        candidate_wt[0],
        model.posterior(torch.tensor([normalize(np.array(candidate_wt).reshape(1, -1), bounds_array)], dtype=torch.double)).mean.item(),
        color="yellow", label="Candidate"
    )
ax.set_xlabel("Carbon Black [wt%]")
ax.set_ylabel("Yield Stress [Pa]")
ax.set_title("GP Prediction")
ax.grid(True)
ax.legend()
st.pyplot(fig)
