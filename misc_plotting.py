import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
  '3 bands': [0.00521, 0.0073, 0.0121],
  '4 bands': [0.00148, 0.0021, 0.00655],
  'Range of days': [1, 2, 3]
}).set_index('Range of days')
df.plot(kind='bar')
plt.show()