# نصب کتابخانه‌ها در صورت نیاز (فقط در Google Colab)
# !pip install pandas numpy matplotlib seaborn scikit-learn

# 1. 📥 وارد کردن کتابخانه‌ها
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 2. 📄 خواندن فایل دیتاست
df = pd.read_csv("C:/Users/Parseh/Desktop/housing.csv")

# 3. 🔍 بررسی اولیه داده‌ها
print(df.head())
print(df.info())
print(df.describe())

# 4. 📊 تحلیل اولیه (EDA)
# فقط ستون‌های عددی برای heatmap
numeric_df = df.select_dtypes(include=["float64", "int"])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# نمودار توزیع قیمت خانه
sns.histplot(df["median_house_value"], kde=True, bins=40, color="green")
plt.title("توزیع قیمت خانه‌ها")
plt.xlabel("قیمت خانه")
plt.ylabel("تعداد")
plt.show()

# Scatter بین درآمد و قیمت خانه
sns.scatterplot(data=df, x="median_income", y="median_house_value", alpha=0.3)
plt.title("درآمد خانوار vs قیمت خانه")
plt.show()

# 5. 🧹 پیش‌پردازش داده‌ها
# پر کردن مقادیر گمشده
df.fillna(df.mean(numeric_only=True), inplace=True)

# انتخاب ویژگی‌ها و هدف
features = ['median_income', 'housing_median_age', 'total_rooms']
target = 'median_house_value'

X = df[features]
y = df[target]

# 6. ✂️ تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 🤖 ساخت مدل
model = LinearRegression()
model.fit(X_train, y_train)

# 8. 📈 پیش‌بینی و ارزیابی
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("📊 Root Mean Squared Error (RMSE):", rmse)

# 9. 📉 مصورسازی نتایج
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # خط y = x
plt.show()
