import pandas as pd
import numpy as np

from sklearn import datasets

from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from evidently.tests import TestNumberOfMissingValues, TestNumberOfUniqueValues, TestShareOfMissingValues, TestValueRange, TestColumnQuantile, TestConflictTarget, TestHighlyCorrelatedColumns, TestColumnAllConstantValues, TestShareOfColumnsWithMissingValues, TestShareOfRowsWithMissingValues, TestNumberOfDifferentMissingValues, TestNumberOfConstantColumns, TestNumberOfEmptyRows, TestNumberOfEmptyColumns, TestNumberOfDuplicatedRows, TestNumberOfDuplicatedColumns, TestColumnsType

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset

iris_frame = datasets.load_iris(as_frame='auto').frame
current_data = pd.read_csv('s8_monitoring/exercise_files/prediction_database.csv', sep=',')
current_data.drop(columns=['time'], inplace=True)
iris_frame.columns = current_data.columns

current_data.iloc[3,1] = np.NaN

data_drift_report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    TargetDriftPreset(),
])

data_drift_report.run(current_data=current_data, reference_data=iris_frame, column_mapping=None)
data_drift_report.save_html('test.html')

data_stability = TestSuite(tests=[
    TestNumberOfMissingValues(),
    DataStabilityTestPreset(),
])
data_stability.run(current_data=current_data, reference_data=iris_frame, column_mapping=None)
data_stability.save_html('test2.html')

tests = TestSuite(tests=[
    TestNumberOfUniqueValues(column_name=' sepal_length'),
    TestValueRange(column_name=' sepal_width'),
    TestColumnQuantile(column_name=' sepal_length', quantile=0.25),
    TestHighlyCorrelatedColumns(),
    TestColumnAllConstantValues(column_name=' petal_length'),
    TestConflictTarget(),
    TestShareOfMissingValues(lt=0.2),
    TestShareOfColumnsWithMissingValues(),
    TestShareOfRowsWithMissingValues(),
    TestNumberOfDifferentMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfEmptyRows(),
    TestNumberOfEmptyColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
])
tests.run(current_data=current_data, reference_data=iris_frame, column_mapping=None)
tests.save_html('test3.html')