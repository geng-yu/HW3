import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import streamlit as st

# Step 1: Streamlit UI
st.title("3D Scatter Plot with Adjustable Parameters")

# Slider for distance threshold
threshold = st.slider("Select distance threshold:", 1.0, 10.0, 4.0)

# Sliders for semi-major axis and semi-minor axis
semi_major_axis = st.slider("Select semi-major axis:", 1.0, 10.0, 6.0)
semi_minor_axis = st.slider("Select semi-minor axis:", 1.0, 10.0, 4.0)

# Generate random points
num_points = 600

# Generate points with elliptical distribution
theta = np.random.uniform(0, 2 * np.pi, num_points)
r = np.sqrt(np.random.uniform(0, 1, num_points))  # Uniformly distributed in a circle
x = r * np.cos(theta) * semi_major_axis
y = r * np.sin(theta) * semi_minor_axis

points = np.column_stack((x, y))

# Calculate distances from the origin
distances = np.linalg.norm(points, axis=1)

# Assign labels based on user-defined threshold
labels = np.where(distances < threshold, 0, 1)

# Step 2: Calculate x3 as a Gaussian function of x1 and x2
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

x3 = gaussian_function(points[:, 0], points[:, 1])

# Step 3: Train a LinearSVC to find a separating hyperplane
X = np.column_stack((points, x3))
labels = labels.astype(int)  # 確保 labels 是整數類型

# 確保 X 和 labels 形狀一致
st.write("Shapes of X and labels:", X.shape, labels.shape)

if np.any(np.isnan(X)) or np.any(np.isnan(labels)):
    st.error("Data contains NaN values.")
else:
    # 確保 labels 有至少兩類
    if len(np.unique(labels)) < 2:
        st.error("Not enough unique labels to fit the model.")
    else:
        clf = LinearSVC(random_state=0, max_iter=10000)
        try:
            clf.fit(X, labels)
            coef = clf.coef_[0]
            intercept = clf.intercept_[0]
        except ValueError as e:
            st.error(f"Error during model fitting: {e}")
            st.stop()  # 停止執行

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[labels == 0][:, 0], points[labels == 0][:, 1], x3[labels == 0], c='blue', marker='o', label='Y=0')
        ax.scatter(points[labels == 1][:, 0], points[labels == 1][:, 1], x3[labels == 1], c='red', marker='s', label='Y=1')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        ax.set_title('3D Scatter Plot with Adjustable Parameters')
        ax.legend()

        # Create a meshgrid to plot the separating hyperplane
        xx, yy = np.meshgrid(np.linspace(min(points[:, 0]), max(points[:, 0]), 10),
                             np.linspace(min(points[:, 1]), max(points[:, 1]), 10))
        zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

        # Plot the hyperplane
        ax.plot_surface(xx, yy, zz, color='lightblue', alpha=0.5)

        # Display the equation of the hyperplane
        equation = f'Hyperplane: {coef[0]:.2f} * x1 + {coef[1]:.2f} * x2 + {intercept:.2f} = 0'
        ax.text2D(0.05, 0.95, equation, transform=ax.transAxes)

        # Show the plot in Streamlit
        st.pyplot(fig)

