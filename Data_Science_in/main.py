# This work follows the Linkdin course "Introduction to Data Science"
# https://www.linkedin.com/learning/introduction-to-data-science-2/next-steps?autoAdvance=true&autoSkip=true&autoplay=true&resume=false&u=41910388


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


def pop_names_2014(us_baby):
    """
    A simple function that takes the US baby data frame and extracts all the 5 most popular names of babies born in 2014.
    :param us_baby: The US baby csv file.
    :return: A list of the 5 most popular names in 2014.
    """
    us_baby_2014 = us_baby.loc[us_baby['Year'] == 2014, :]
    sorted_us_baby_2014 = us_baby_2014.sort_values('Count', ascending=False)
    return sorted_us_baby_2014.iloc[0:5]

def pop_names_cali_2014(state_baby):
    """
    A simple function that takes the US state baby data frame and extracts all the 5 most popular names of babies born
    in 2014 in California.
    :param state_baby: The US state baby csv file.
    :return: A list of the 5 most popular names in 2014.
    """
    state_baby_2014 = state_baby.loc[state_baby['Year'] == 2014, :]
    state_baby_2014_cali = state_baby_2014.loc[state_baby_2014['State'] == 'CA', :]
    sorted_state_baby_2014_cali = state_baby_2014_cali.sort_values('Count', ascending=False)
    return sorted_state_baby_2014_cali.iloc[0:5]


def popular(pd_series):
    """
    A function that receives a panda series containing baby names by descending order and returns the most popular baby
    names.
    :param pd_series: A pandas series.
    :return: The most popular baby names.
    """
    return pd_series.iloc[0]


def pop_name_cali_gender_year(state_baby):
    """
    A simple function that takes the US state baby data frame and extracts all the most popular names of babies born
    in California for male and female each year.
    :param state_baby: The US state baby csv file.
    :return: A list of the most popular names for male and female in California in each year.
    """
    state_baby_cali = state_baby.loc[state_baby['State'] == 'CA', :]
    state_baby_cali_group = state_baby_cali.sort_values('Count', ascending=False).groupby(['Year', 'Gender']).agg(popular)
    return state_baby_cali_group


def my_name_plot(us_baby):
    """
    A simple function that shows in a graph the amount of babies that were named Doron in each year.
    :param us_baby: The US baby csv file.
    """
    us_name = us_baby.loc[us_baby['Name'] == 'Doron', :]
    us_name.plot.barh(x='Year', y='Count')
    plt.show()


def incident_crime(crime):
    """
    A simple function that takes the Boston crime data frame, and returns the data frame grouped by incident number and
    counts.
    :param crime: The data frame for crime in Boston.
    :return: The data frame grouped by incident number and counts.
    """
    return crime.groupby(['INCIDENT_NUMBER']).count()


def missing(crime):
    """
    A simple function that returns a new data frame with all the missing values.
    :param crime: The data frame for crime in Boston.
    :return: A data frame where all missing values are true.
    """
    return crime.isnull()


def miss_row(crime):
    """
    A simple function that returns a new data frame with all the rows that have a missing values.
    :param crime: The data frame for crime in Boston.
    :return: A series of all the rows that have a missing value.
    """
    return crime.isnull().any(axis=1)


def get_miss(crime):
    """
    A simple function that returns a data frame with all the rows with missing values in the data frame.
    :param crime: The data frame for crime in Boston.
    :return: The data frame with all the rows with missing values in the data frame.
    """
    return crime[crime.isnull().any(axis=1)]


def drop_time(crime):
    """
    A simple function that returns the data frame without time columns.
    :param crime: The data frame for crime in Boston.
    :return: The data frame without time columns.
    """
    return crime.drop(columns=['YEAR', 'MONTH', 'HOUR'])


def misspelling(cleaned):
    """
    A function that prints all the unique values to check for misspelling.
    :param cleaned: The cleaned data frame for crime in Boston.
    """
    print(cleaned['OFFENSE_CODE_GROUP'].unique)
    print(cleaned['OFFENSE_DESCRIPTION'].unique)
    print(cleaned['DAY_OF_WEEK'].unique)


def create_plot(listings):
    """
    A simple function that creates a plot of the amount of listings per neighbourhood.
    :param listings: A data frame.
    """
    sn.countplot(x='neighbourhood_group', data=listings)
    plt.show()


def bar_plot(listings):
    """
    A simple function that creates a plot of the price of listings per neighbourhood.
    :param listings: A data frame.
    """
    sn.barplot(x='neighbourhood_group', y='price', data=listings)  # With confidence intervals.
    sn.barplot(x='neighbourhood_group', y='price', data=listings, ci=False)
    plt.show()


def hist_plot(listings):
    """
    A simple function that creates a histogram.
    :param listings: A data frame.
    """
    plt.hist(listings['price'])
    plt.xlabel('price (in US dollars)')
    plt.show()


def hist2(listings):
    """
    A simple function that displays a histogram in a more spacious way.
    :param listings: A data frame.
    """
    plt.hist(listings['price'], bins=np.arange(0, 1100, 40))
    plt.xlabel('price (in US dollars)')
    plt.show()


def scatter(listings):
    """
    A simple function that creates a scatter plot.
    :param listings: A data frame.
    """
    plt.scatter(x=listings['price'], y=listings['number_of_reviews'])
    plt.xlabel('price')
    plt.ylabel('number of reviews')
    plt.show()


def lim_scatter(listings):
    """
    A simple function that creates a scatter plot which is limited in the x-axis, and the size of the points is smaller.
    :param listings: A data frame.
    """
    plt.scatter(x=listings['price'], y=listings['number_of_reviews'], s=5)
    plt.xlabel('price')
    plt.ylabel('number of reviews')
    plt.xlim(0, 1100)
    plt.show()


def hyp_test(avocado_info):
    """
    A simple function that helps to run a hypothesis test.
    :param avocado_info: A data frame contating the number of days it took trees to grow and true or false if they have
    been fertilized.
    """
    fertilized = avocado_info.loc[avocado_info['Fertilizer'] == True]
    not_fertilized = avocado_info.loc[avocado_info['Fertilizer'] == False]
    sn.distplot(fertilized['Growth Duration'], kde=False, label='Fertilized')
    sn.distplot(not_fertilized['Growth Duration'], kde=False, label='Not Fertilized')
    plt.legend()
    plt.show()


def simple_perm(avocado_info):
    """
    A simple function that helps to run a permutation test.
    :param avocado_info: A data frame containing the number of days it took trees to grow and true or false if they have
    been fertilized.
    """
    fertilized = avocado_info.loc[avocado_info['Fertilizer'] == True]
    not_fertilized = avocado_info.loc[avocado_info['Fertilizer'] == False]
    observed_test_stat = np.mean(fertilized['Growth Duration'] - np.mean(not_fertilized['Growth Duration']))
    print(observed_test_stat)
    print(avocado_info['Growth Duration'].sample(frac=1).reset_index())  # Create permutation and reset indexes.


def perm(data):
    """
    A simple generic function that creates a permutation, without indexes
    :param data: A given data frame.
    :return: A permuted data frame.
    """
    return data.sample(frac=1).reset_index(drop=True)


def perm_test(avocado_info):
    """
    A function that runs a permutation test and prints the p-value.
    :param avocado_info: A data frame containing the number of days it took trees to grow and true or false if they have
    been fertilized.
    """
    sim_test_stat = np.array([])
    reps = 10000
    for i in range(reps):
        perm_info = perm(avocado_info['Growth Duration'])
        df = pd.DataFrame({'Permuted Duration': perm_info, 'Fertilizer': avocado_info['Fertilizer']})
        fertilized = df.loc[df['Fertilizer'] == True, 'Permuted Duration']
        not_fertilized = df.loc[df['Fertilizer'] == False, 'Permuted Duration']
        stat = np.mean(fertilized) - np.mean(not_fertilized)
        sim_test_stat = np.append(sim_test_stat, stat)
    print(sim_test_stat)  # Sanity check.
    o_fertilized = avocado_info.loc[avocado_info['Fertilizer'] == True]
    o_not_fertilized = avocado_info.loc[avocado_info['Fertilizer'] == False]
    observed_test_stat = np.mean(o_fertilized['Growth Duration'] - np.mean(o_not_fertilized['Growth Duration']))
    p_value = np.count_nonzero(sim_test_stat <= observed_test_stat) / reps
    print(p_value)


def run_bootstrap(avocado_info):
    """
    The main method to use to run a bootstrap to find a confidence interval.
    :param avocado_info: A data frame containing the number of days it took trees to grow and true or false if they have
    been fertilized.
    """
    fertilized = avocado_info.loc[avocado_info['Fertilizer'] == True, 'Growth Duration']
    not_fertilized = avocado_info.loc[avocado_info['Fertilizer'] == False, 'Growth Duration']
    fertilized_means = bootstrap(fertilized, 10000)
    not_fertilized_means = bootstrap(not_fertilized, 10000)
    estimates = fertilized_means - not_fertilized_means
    sn.distplot(estimates, kde=False)
    plt.show()
    print(np.percentile(estimates, 2.5), np.percentile(estimates, 97.5))


def resample(orig_sample):
    """
    A simple function to help with resampling.
    :param orig_sample: The original sample.
    :return: A randomly generated sample.
    """
    return np.random.choice(orig_sample, size=len(orig_sample))


def bootstrap(orig_sample, reps):
    """
    A function that uses the original samples to resample and get means into an array and returns it.
    :param orig_sample: The original sample.
    :param reps: The number of repetitions.
    :return: An array of all the means that we got while resampling.
    """
    means = np.array([])
    for i in range(reps):
        new_sample = resample(orig_sample)
        new_mean = np.mean(new_sample)
        means = np.append(means, new_mean)
    return means


if __name__ == '__main__':
    # Work using the US baby names file.
    us_baby = pd.read_csv("us_baby_names.csv")
    print(us_baby)
    print(pop_names_2014(us_baby))
    my_name_plot(us_baby)

    # Work using the US state baby names file.
    state_baby = pd.read_csv("state_baby_names.csv")
    print(state_baby)
    print(pop_names_cali_2014(state_baby))
    print(pop_name_cali_gender_year(state_baby))

    # Work using the Boston crime file.
    crime = pd.read_csv("crime_boston.csv")
    print(incident_crime(crime))
    print(missing(crime))
    print(miss_row(crime))
    print(get_miss(crime))
    crime_cleaned = drop_time(crime)
    misspelling(crime_cleaned)
    crime_cleaned = crime_cleaned.drop(columns='Location')

    # Work using the Airbnb file.
    listing = pd.read_csv("Airbnb_NYC_2019.csv")
    create_plot(listing)
    bar_plot(listing)
    hist_plot(listing)
    hist2(listing)
    scatter(listing)
    lim_scatter(listing)

    # Work using the avocado file.
    avocado_info = pd.read_csv("avocado_info.csv")
    hyp_test(avocado_info)
    simple_perm(avocado_info)
    perm_test(avocado_info)
    run_bootstrap(avocado_info)



