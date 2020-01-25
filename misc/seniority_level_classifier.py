import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import logging
from model_training.resume_matching_utils.custom_preprocessing import CustomPreprocessing
from model_training.resume_matching_utils.file_loader import get_default_config, get_resource_path
from model_training.resume_matching_utils.data_i_o import data_i_o

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class SeniorityLevelClassifier:

    def __init__(self, config=None):
        if config is None:
            self.config = get_default_config()
        else:
            self.config = config
        job_matches_filepath = get_resource_path(self.config["data_folder"],
                                                 self.config["job_matches_filepath_seniority"])
        dio = data_i_o()
        df_req = dio.get_requisition()
        self.job_matches = df_req
        #self.job_matches = pd.read_csv(job_matches_filepath)
        self._clean_job_matches()
        df_res = dio.get_resume()
        #resumes_filepath = get_resource_path(self.config["data_folder"], self.config["resumes_filepath"])
        #self.resumes = pd.read_csv(resumes_filepath)
        self.resumes = df_res
        self._clean_resumes()
        self.X_train, self.X_test, self.Y_train, self.Y_test = self._create_train_and_test_set_for_resumes()
        self.merged_dataset = self._merge_datasets()
        self.X_train_all = self.merged_dataset['text']
        self.Y_train_all = self.merged_dataset['label']

    def _clean_job_matches(self):
        logger.info("Shape of job matches dataframe: {}. Filtering...".format(self.job_matches.shape))
        self.job_matches.fillna('')
        self.job_matches = self.job_matches[self.job_matches['RequisitionStatus'] == 'Filled']
        self.job_matches['TargetStartDate_dt'] = pd.to_datetime(self.job_matches['TargetStartDate'])
        self.job_matches = self.job_matches[self.job_matches['TargetStartDate_dt'] > datetime.date(2016, 1, 1)]
        self.job_matches['JobDescription'] = self.job_matches['InternalJobDescription'] + ' ' + self.job_matches[
            'InternalJobQualification']
        self.job_matches['JobDescription'].fillna('')
        self.job_matches['label'] = self.job_matches['CareerLevelDesc'].apply(CustomPreprocessing.career_label)
        self.job_matches['JobDescription'] = self.job_matches['JobDescription'].apply(CustomPreprocessing.clean)
        logger.info("Finished filtering. Shape of job matches dataframe: {}".format(self.job_matches.shape))

    def _clean_resumes(self):
        logger.info("Shape of resumes dataframe: {}. Cleaning resumes...".format(self.resumes.shape))
        self.resumes.fillna('')
        self.resumes['resume_txt_clean'] = self.resumes['resume_txt'].apply(CustomPreprocessing.clean)
        self.resumes['resume_txt_clean'].fillna('')
        self.resumes['label'] = self.resumes['CareerLevelDesc'].apply(CustomPreprocessing.career_label)
        logger.info("Finished cleaning. Shape of resumes dataframe: {}".format(self.resumes.shape))

    def _create_train_and_test_set_for_resumes(self):
        logger.info("Splitting dataset to train and test...")
        X = self.resumes['resume_txt_clean']
        Y = self.resumes['label']
        X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(X, Y, self.resumes.index,
                                                                                         test_size=0.33, random_state=0)
        return X_train, X_test, Y_train, Y_test

    def _merge_datasets(self):
        logger.info("Merging job matches and resume dataframes...")
        merged_dataset = pd.DataFrame(columns=['text', 'label'])
        merged_dataset['text'] = self.X_train
        merged_dataset['label'] = self.Y_train
        merged_dataset = pd.concat([merged_dataset, self.job_matches.rename(columns={'JobDescription': 'text'})],
                                   ignore_index=True)
        merged_dataset = merged_dataset[pd.notnull(merged_dataset['text'])]
        logging.info("Created merged dataset with shape: {}".format(merged_dataset.shape))
        return merged_dataset

    def train_seniority_level_classifier(self, test=True):
        logger.info("Training seniority level classifier...")
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=CustomPreprocessing.custom_tokenizer_for_seniority_classifier)),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(random_state=0, class_weight='balanced'))])
        pipeline.fit(self.X_train_all, self.Y_train_all)
        # with open(SENIORITY_CLASSIFIER_FILEPATH, 'wb') as f:
        #     dump(pipeline, f)
        # saved_model = joblib.load(SENIORITY_CLASSIFIER_FILEPATH)
        classifier_filepath = get_resource_path(self.config["models_folder"],
                                                self.config["seniority_level_classifier_latest_filename"])
        logger.info("Trained model. Saving to pickle file: {}".format(classifier_filepath))
        with open(classifier_filepath, 'wb') as f2:
            # pickle.dump(saved_model, f2)
            pickle.dump(pipeline, f2)  # TODO: test
        if test:
            Y_predicted = pipeline.predict(self.X_test)
            logger.info("Classification report on held out test set: {}".format(
                (classification_report(self.Y_test, Y_predicted))))
