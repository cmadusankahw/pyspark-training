from pandas import DataFrame
import matplotlib.pyplot as plt

Data = {'Unemployment_Rate': [6.1, 5.8, 5.7, 5.7, 5.8, 5.6, 5.5, 5.3, 5.2, 5.2],
        'Stock_Index_Price': [1500, 1520, 1525, 1523, 1515, 1540, 1545, 1560, 1555, 1565]
        }

df = DataFrame(Data, columns=['Unemployment_Rate', 'Stock_Index_Price'])

# scatter graph
df.plot(x='Unemployment_Rate', y='Stock_Index_Price', kind='scatter')
plt.show()

# line graph
df.plot(x ='Year', y='Unemployment_Rate', kind = 'line')
plt.show()

Data2 = {'Tasks': [300, 500, 700]}
df2 = DataFrame(Data, columns=['Tasks'], index=['Tasks Pending', 'Tasks Ongoing', 'Tasks Completed'])

# pie chart
df.plot.pie(y='Tasks', figsize=(5, 5), autopct='%1.1f%%', startangle=90)
plt.show()


# saving generated graph
# plt.savefig("graph1.png")