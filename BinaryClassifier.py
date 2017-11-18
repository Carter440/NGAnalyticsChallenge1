from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import tensorflow as tf
import pandas as pd
import numpy as np
import random
from copy import deepcopy

tf.logging.set_verbosity(tf.logging.ERROR)
#io
testFile = "AnalyticsChallenge1-Testing.csv"
resultFile = "AnalyticsChallenge1-Results.csv"
outputFile = "AnalyticsChallenge1-Prediction"
negFile = "AnalyticsChallenge1-Negs.csv"
posFile = "AnalyticsChallenge1-Pos.csv"

COLUMNS = ["Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField",
    "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "OverTime",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears","TrainingTimesLastYear",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager"]

CATEGORICAL_COLUMNS = ["BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime"]
CONTINUOUS_COLUMNS = ["Age", "DailyRate", "Education", "EnvironmentSatisfaction",
    "HourlyRate", "DistanceFromHome", "JobInvolvement", "JobLevel", "JobSatisfaction",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike",
    "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
     "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]
#needed to compute noise later on (min,max)
CONTINUOUS_RANGES = {
    "Age": (18,60),
    "DailyRate": (102,1499),
    "Education": (1,5),
    "EnvironmentSatisfaction": (1,4),
    "HourlyRate": (30,100),
    "DistanceFromHome": (1,29),
    "JobInvolvement": (1,4),
    "JobLevel": (1,5),
    "JobSatisfaction": (1,4),
    "MonthlyIncome": (1009,19999),
    "MonthlyRate": (2094,26999),
    "NumCompaniesWorked": (0,9),
    "PercentSalaryHike": (11,25),
    "PerformanceRating": (3,4),
    "RelationshipSatisfaction": (1,4),
    "StockOptionLevel": (0,3),
    "TotalWorkingYears": (0,40),
    "TrainingTimesLastYear": (0,6),
    "WorkLifeBalance": (1,4),
    "YearsAtCompany": (0,37),
    "YearsInCurrentRole": (0,18),
    "YearsSinceLastPromotion": (0,15),
    "YearsWithCurrManager": (0,17),
}
#Constants
LABEL_COLUMN = "label"
modelType = "Combined"
USECROSSEDCOLLUMNS = True
ITERATIONS = 10
USEALL = False
THRESHOLD = 0.5
#hyper-prams
trainSteps = 3000
deepLearnRate = 0.001
wideLearnRate = 0.001
dropRate = 0.25
jitteRate = 0.3
maxJitter = (1,8) #Fractional representation (numerator, denominator)
netShape = [128, 64]
#inputFn takes dataframe and returns un-altered features and labels
def inputFn(df):
    continuousCols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in CONTINUOUS_COLUMNS}
    categoricalCols = {k: tf.SparseTensor(indices=[[i,0] for i in range(df[k].size)],
    values = df[k].values,
    dense_shape=[df[k].size, 1])
    for k in CATEGORICAL_COLUMNS}
    featureCols = dict(continuousCols)
    featureCols.update(categoricalCols)
    label = tf.constant(df[LABEL_COLUMN].values)
    return featureCols, label
#helper function to jitterFn
def applyJitter(x, col):
    if random.random() < jitteRate:
        return (random.randint(CONTINUOUS_RANGES[col][0], CONTINUOUS_RANGES[col][1]) + (x*(maxJitter[1] - maxJitter[0])))//maxJitter[1]
    else:
        return x
#inputFn + noise, takes dataframe and returns altered features with origional labels
def jitterFn(df):
    for k in CONTINUOUS_COLUMNS:
        df[k] = df[k].map(lambda x: applyJitter(x, k))
    continuousCols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in CONTINUOUS_COLUMNS}
    categoricalCols = {k: tf.SparseTensor(indices=[[i,0] for i in range(df[k].size)],
    values = df[k].values,
    dense_shape=[df[k].size, 1])
    for k in CATEGORICAL_COLUMNS}
    featureCols = dict(continuousCols)
    featureCols.update(categoricalCols)
    label = tf.constant(df[LABEL_COLUMN].values)
    return featureCols, label
#like inputFn but only produces features
def predictFn(df):
    continuousCols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in CONTINUOUS_COLUMNS}
    categoricalCols = {k: tf.SparseTensor(indices=[[i,0] for i in range(df[k].size)],
    values = df[k].values,
    dense_shape=[df[k].size, 1])
    for k in CATEGORICAL_COLUMNS}
    featureCols = dict(continuousCols)
    featureCols.update(categoricalCols)
    return featureCols
#returns model obj
def buildEstimator(modelDir):
    #construct categoricalCols
    businesstravel = tf.contrib.layers.sparse_column_with_keys(column_name="BusinessTravel",
        keys=["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    department = tf.contrib.layers.sparse_column_with_keys(column_name="Department",
        keys=["Research & Development", "Sales", "Human Resources"])
    educationfield = tf.contrib.layers.sparse_column_with_keys(column_name="EducationField",
        keys=["Technical Degree", "Life Sciences", "Medical", "Marketing",
        "Human Resources", "Other"])
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="Gender",
        keys=["Male", "Female"])
    jobrole = tf.contrib.layers.sparse_column_with_keys(column_name="JobRole",
        keys=["Research Scientist", "Sales Representative", "Sales Executive",
        "Laboratory Technician", "Research Director", "Manufacturing Director",
        "Healthcare Representative", "Human Resources", "Manager"])
    maritalstatus = tf.contrib.layers.sparse_column_with_keys(column_name="MaritalStatus",
        keys=["Married", "Single", "Divorced"])
    overtime = tf.contrib.layers.sparse_column_with_keys(column_name="OverTime",
        keys=["Yes", "No"])

    #construct continuous columns
    age = tf.contrib.layers.real_valued_column("Age")
    dailyrate = tf.contrib.layers.real_valued_column("DailyRate")
    education = tf.contrib.layers.real_valued_column("Education")
    environmentsatisfaction = tf.contrib.layers.real_valued_column("EnvironmentSatisfaction")
    hourlyrate = tf.contrib.layers.real_valued_column("HourlyRate")
    distancefromhome = tf.contrib.layers.real_valued_column("DistanceFromHome")
    jobinvolvement = tf.contrib.layers.real_valued_column("JobInvolvement")
    joblevel = tf.contrib.layers.real_valued_column("JobLevel")
    jobsatisfaction = tf.contrib.layers.real_valued_column("JobSatisfaction")
    monthlyincome = tf.contrib.layers.real_valued_column("MonthlyIncome")
    monthlyrate = tf.contrib.layers.real_valued_column("MonthlyRate")
    numcompaniesworked = tf.contrib.layers.real_valued_column("NumCompaniesWorked")
    percentsalaryhike = tf.contrib.layers.real_valued_column("PercentSalaryHike")
    performancerating = tf.contrib.layers.real_valued_column("PerformanceRating")
    relationshipsatisfaction = tf.contrib.layers.real_valued_column("RelationshipSatisfaction")
    stockoptionlevel = tf.contrib.layers.real_valued_column("StockOptionLevel")
    totalworkingyears = tf.contrib.layers.real_valued_column("TotalWorkingYears")
    trainingtimeslastyear = tf.contrib.layers.real_valued_column("TrainingTimesLastYear")
    worklifebalance = tf.contrib.layers.real_valued_column("WorkLifeBalance")
    yearsatcompany = tf.contrib.layers.real_valued_column("YearsAtCompany")
    yearsincurrentrole = tf.contrib.layers.real_valued_column("YearsInCurrentRole")
    yearssincelastpromotion = tf.contrib.layers.real_valued_column("YearsSinceLastPromotion")
    yearswithcurrmanager = tf.contrib.layers.real_valued_column("YearsWithCurrManager")
#bucketize continuous columns
    ageBuckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60])
    workYearsBuckets = tf.contrib.layers.bucketized_column(totalworkingyears, boundaries=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40])
    wageBuckets = tf.contrib.layers.bucketized_column(hourlyrate, boundaries=[30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    distanceBuckets = tf.contrib.layers.bucketized_column(distancefromhome, boundaries=[0, 2, 6, 10, 15, 20, 25, 30])
    incomeBuckets = tf.contrib.layers.bucketized_column(monthlyincome, boundaries=[1000,2000,4000,6000,8000,10000,12000,14000,16000,18000,20000])
    dailyBuckets = tf.contrib.layers.bucketized_column(dailyrate, boundaries=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500])
    monthlyBuckets = tf.contrib.layers.bucketized_column(monthlyrate, boundaries=[2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 27000])
    compyearsBuckets = tf.contrib.layers.bucketized_column(yearsatcompany,boundaries=[0,2,4,6,8,10,15,20,25,30,40])
    roleyearsBuckets = tf.contrib.layers.bucketized_column(yearsincurrentrole,boundaries=[0,2,4,6,8,10,12,14,16,20])
    promoyearsBuckets = tf.contrib.layers.bucketized_column(yearssincelastpromotion,boundaries=[0,1,2,4,6,8,12,16])
    managerBuckets = tf.contrib.layers.bucketized_column(yearswithcurrmanager,boundaries=[0,2,4,6,8,10,12,14,16,18])
    raiseBuckets = tf.contrib.layers.bucketized_column(percentsalaryhike,boundaries=[10,12,14,16,18,20,22,24,26])
    educationBuckets = tf.contrib.layers.bucketized_column(education, boundaries=[1,2,3,4,5])
    environmentBuckets = tf.contrib.layers.bucketized_column(environmentsatisfaction, boundaries=[1,2,3,4])
    jobinvolvementBuckets = tf.contrib.layers.bucketized_column(jobinvolvement, boundaries=[1,2,3,4])
    joblevelBuckets = tf.contrib.layers.bucketized_column(joblevel, boundaries=[1,2,3,4,5])
    jobsatisfactionBuckets = tf.contrib.layers.bucketized_column(jobsatisfaction, boundaries=[1,2,3,4])
    numcompaniesworkedBuckets = tf.contrib.layers.bucketized_column(numcompaniesworked, boundaries=[0,1,2,3,4,5,6,7,8,9])
    performanceBuckets = tf.contrib.layers.bucketized_column(performancerating, boundaries=[3,4])
    relationshipBuckets = tf.contrib.layers.bucketized_column(relationshipsatisfaction, boundaries=[1,2,3,4])
    stockoptionBuckets = tf.contrib.layers.bucketized_column(stockoptionlevel, boundaries=[0,1,2,3])
    trainingtimesBuckets = tf.contrib.layers.bucketized_column(trainingtimeslastyear, boundaries=[0,1,2,3,4,5,6])
    worklifeBuckets = tf.contrib.layers.bucketized_column(worklifebalance, boundaries=[1,2,3,4])
    wideCols = []
    deepCols = []
#use all data
    if USEALL:
        wideCols = [businesstravel, department, educationfield, gender, jobrole,
            maritalstatus, overtime, ageBuckets, workYearsBuckets, wageBuckets,
            distanceBuckets, incomeBuckets, dailyBuckets, monthlyBuckets,
            compyearsBuckets, roleyearsBuckets, managerBuckets, raiseBuckets,
            worklifeBuckets, trainingtimesBuckets, relationshipBuckets,
            performanceBuckets, stockoptionBuckets, numcompaniesworkedBuckets,
            jobsatisfactionBuckets, joblevelBuckets,jobinvolvementBuckets,
            environmentBuckets, educationBuckets]
        deepCols = [tf.contrib.layers.embedding_column(businesstravel, dimension=2),
            tf.contrib.layers.embedding_column(department, dimension=2),
            tf.contrib.layers.embedding_column(educationfield, dimension=3),
            tf.contrib.layers.embedding_column(gender, dimension=1),
            tf.contrib.layers.embedding_column(jobrole, dimension=4),
            tf.contrib.layers.embedding_column(maritalstatus, dimension=2),
            tf.contrib.layers.embedding_column(overtime, dimension=1),
            age, dailyrate, education, environmentsatisfaction,
            hourlyrate, distancefromhome, jobinvolvement, joblevel, jobsatisfaction,
            monthlyincome, monthlyrate, numcompaniesworked, percentsalaryhike,
            performancerating, relationshipsatisfaction, stockoptionlevel, totalworkingyears,
            trainingtimeslastyear, worklifebalance, yearsatcompany, yearsincurrentrole,
            yearssincelastpromotion, yearswithcurrmanager]
    else:
    #cut out insignificant data
        wideCols = [businesstravel, department, educationfield, gender,
            maritalstatus, overtime, ageBuckets,
            distanceBuckets,  dailyBuckets, workYearsBuckets,
            compyearsBuckets, roleyearsBuckets, managerBuckets,
            stockoptionBuckets, relationshipBuckets, numcompaniesworkedBuckets,
            jobsatisfactionBuckets, joblevelBuckets,
            environmentBuckets, educationBuckets]
        deepCols = [tf.contrib.layers.embedding_column(businesstravel, dimension=2),
            tf.contrib.layers.embedding_column(department, dimension=2),
            tf.contrib.layers.embedding_column(educationfield, dimension=3),
            tf.contrib.layers.embedding_column(gender, dimension=1),
            tf.contrib.layers.embedding_column(maritalstatus, dimension=2),
            tf.contrib.layers.embedding_column(overtime, dimension=1),
            age, dailyrate, education, environmentsatisfaction,
            distancefromhome, joblevel, jobsatisfaction,
            numcompaniesworked,
            relationshipsatisfaction, stockoptionlevel, totalworkingyears,
            yearsatcompany, yearsincurrentrole,
            yearswithcurrmanager]
    #utilize crossed columns for linear classification
    if USECROSSEDCOLLUMNS:
        wideCols.extend([
        tf.contrib.layers.crossed_column([gender, educationfield, ageBuckets, incomeBuckets], hash_bucket_size= int(1e6)),
        tf.contrib.layers.crossed_column([incomeBuckets, dailyBuckets], hash_bucket_size= int(1e4)),
        tf.contrib.layers.crossed_column([distanceBuckets, environmentBuckets, relationshipBuckets], hash_bucket_size= int(1e4)),
        tf.contrib.layers.crossed_column([joblevelBuckets, jobsatisfactionBuckets, roleyearsBuckets], hash_bucket_size= int(1e6)),
        tf.contrib.layers.crossed_column([ageBuckets, workYearsBuckets, compyearsBuckets, managerBuckets], hash_bucket_size= int(1e6)),
        tf.contrib.layers.crossed_column([gender, maritalstatus, relationshipBuckets], hash_bucket_size= int(1e4)),
        tf.contrib.layers.crossed_column([incomeBuckets, dailyBuckets, stockoptionBuckets, overtime], hash_bucket_size= int(1e6)),
        tf.contrib.layers.crossed_column([businesstravel, overtime, environmentBuckets], hash_bucket_size= int(1e4)),
        tf.contrib.layers.crossed_column([ageBuckets, joblevelBuckets, incomeBuckets, stockoptionBuckets], hash_bucket_size= int(1e4))
        ])
    #combined wide and deep learning network
    if modelType == "Combined":
        return tf.contrib.learn.DNNLinearCombinedClassifier(model_dir=modelDir,
            linear_feature_columns=wideCols, dnn_feature_columns=deepCols,
            dnn_hidden_units=netShape, dnn_optimizer=tf.train.AdamOptimizer(deepLearnRate),
            linear_optimizer=tf.train.FtrlOptimizer(
            learning_rate=wideLearnRate,
            l1_regularization_strength=1.0,
            l2_regularization_strength=1.0),
            dnn_activation_fn=tf.nn.relu,
            dnn_dropout=dropRate, fix_global_step_increment_bug=True)
    #just deep neural-net
    elif modelType == "Neural":
        return tf.contrib.learn.DNNClassifier(netShape, deepCols,
            dropout= dropRate, optimizer=tf.train.AdamOptimizer(deepLearnRate))
    #just linear classifier
    else:
        return tf.contrib.learn.LinearClassifier(wideCols,
            optimizer=tf.train.FtrlOptimizer(learning_rate=wideLearnRate,
                        l1_regularization_strength=1.0,
                        l2_regularization_strength=1.0))
#save results to disk
def recordPredictions(predictions, iteration):
    df_res = pd.read_csv(tf.gfile.Open(resultFile), names=["EmployeeNumber", "Attrition"],
        skipinitialspace=True, skiprows=1,engine="python")
    for i in range(len(df_res)):
        df_res["Attrition"][i] = predictions[df_res["EmployeeNumber"][i]]
    df_res.to_csv(outputFile+str(iteration)+".csv")
#set threshold to reduce false positives
def applyThreshold(thresh, prob):
    if prob > thresh:
        return 1
    return 0
#train model, evaluate it, make predictions on un-classified data
def trainModel(iteration):
    print("="*77)
    print("iteration: {}".format(iteration))
    df_negs = pd.read_csv(tf.gfile.Open(negFile), names=COLUMNS, skipinitialspace=True,
        skiprows=1,engine="python")
    df_negs = df_negs.sample(frac=1).reset_index(drop=True)
    df_pos = pd.read_csv(tf.gfile.Open(posFile), names=COLUMNS, skipinitialspace=True,
        skiprows=1,engine="python")
    df_pos = df_pos.sample(frac=1).reset_index(drop=True)
    df_test = pd.read_csv(tf.gfile.Open(testFile), names=COLUMNS,
        skiprows=1,engine="python")
    df_negs = df_negs.dropna(how='any', axis = 0)
    df_negs[LABEL_COLUMN] = (df_negs["Attrition"].apply(lambda x: "Yes" in x)).astype(int)
    df_pos = df_pos.dropna(how='any', axis = 0)
    df_pos[LABEL_COLUMN] = (df_pos["Attrition"].apply(lambda x: "Yes" in x)).astype(int)
    df_train = df_negs.iloc[:(len(df_negs)//10)*8]
    for _ in range(5):
        df_train = pd.concat([df_train,df_pos.iloc[:(len(df_pos)//10)*8]])
    df_valid = df_negs.iloc[(len(df_negs)//10)*8:]
    for _ in range(5):
        df_valid = pd.concat([df_valid,df_pos.iloc[(len(df_pos)//10)*8:]])
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_valid = df_valid.sample(frac=1).reset_index(drop=True)
    df_test = df_test.dropna(how='any', axis = 0)
    df_test[LABEL_COLUMN] = (df_test["Attrition"].apply(lambda x: "Yes" in x)).astype(int)
    modelDir = tempfile.mkdtemp()
    m = buildEstimator(modelDir)
    m.fit(input_fn=lambda: jitterFn(deepcopy(df_train)), steps=trainSteps)
    results = m.evaluate(input_fn = lambda: inputFn(df_train), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    results = m.evaluate(input_fn = lambda: inputFn(df_valid), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    probs = [k for k in m.predict_proba(input_fn=lambda: predictFn(df_test))]
    attrition = ["No", "Yes"]
    preDict = {df_test["EmployeeNumber"][k]: attrition[applyThreshold(THRESHOLD,probs[k][1])] for k in range(len(probs))}
    recordPredictions(preDict, iteration)


def main(_):
    for i in range(ITERATIONS):
        trainModel(iteration=i)

if __name__ == "__main__":
    tf.app.run(main=main)
