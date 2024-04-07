from mlxtend.preprocessing import TransactionEncoder
import pandas as pd


data = pd.read_csv('Final.csv')
#data = data[[ 'HomeFouls_Discrete', 'AwayFouls_Discrete','Result','HomeTeam','AwayTeam', 'HomeShotsOnTarget_Discrete','AwayShotsOnTarget_Discrete', 'Time']]
data.dropna(inplace=True)


print(len(data))
# Creating a list of transactions with the discretized values
transactions = data[[ 'HomeFouls_Discrete', 'AwayFouls_Discrete','Result','HomeTeam','AwayTeam', 'HomeShotsOnTarget_Discrete','AwayShotsOnTarget_Discrete']].apply(lambda x: '-'.join(x), axis=1).tolist()
transactions = [transaction.split('-') for transaction in transactions]
# Initializing the TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Display the first few rows of the prepared dataset

from mlxtend.frequent_patterns import apriori

# Use the apriori algorithm to find frequent itemsets
# Adjust the min_support parameter as needed
frequent_itemsets = apriori(df, min_support=0.15, use_colnames=True)

# Display the frequent itemsets
print(frequent_itemsets)


from mlxtend.frequent_patterns import association_rules

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)
# Display the rules, sorted by confidence
rules = rules.sort_values(by="confidence", ascending=False)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
