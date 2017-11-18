import pandas
import numpy

LABEL_COLUMN = "label"

fn = "AnalyticsChallenge1-Raw.csv"
names = ["Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField",
    "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "OverTime",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears","TrainingTimesLastYear",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager"]
data = pandas.read_csv(fn, names=names, skiprows=1,engine="python")
data[LABEL_COLUMN] = (data["Attrition"].apply(lambda x: "Yes" in x)).astype(int)
correlations = data.corr()
print(correlations['label'])
