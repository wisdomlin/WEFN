# visualize the data
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# Load dataset
names = [
    'Renewable Water Available per capita', 
    'Active Working Population in Agriculture Percentage', 
    'Net Agricultural Product Import (Import-Export)', 
    'Yearly Population Increase Percentage', 
    'Label']
filepath = "01_RawData/01_DomainKnowledge.csv"
dataset = read_csv(filepath, names=names)

# dataset=dataset.astype(float)
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()