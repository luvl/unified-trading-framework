import matplotlib.pyplot as plt
import pandas as pd 

loc = "/home/linhdn/Developer/unified-framework-for-trading/data/stock-data/vnm-data.csv"

dat = pd.read_csv(loc)

print(dat.head())
print(len(dat['Close']))

plt.figure(figsize=(12,7))
plt.plot(range(1500), dat['Close'][:1500].values, color='b', label="Train set")
plt.plot(range(1500,2500), dat['Close'][1500:2500].values, color='r', label="Test set")
plt.plot(range(2500,len(dat['Close'])-1), dat['Close'][2500:len(dat['Close'])-1].values, color='g', label="Validation set")
plt.legend(loc="upper left")
plt.show()