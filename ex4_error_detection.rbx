#sigma

-- Contracts
@ctgov_study_type:string
@drks_study_type:string
@study_type_non_interventional:string
@observational_model:string
@ctgov_purpose:string
@drks_purpose:string


--Domain constraints
ctgov_study_type notin {'Interventional','Observational','Observational [Patient Registry]'}
drks_study_type notin {'Interventional','Non-interventional'}
study_type_non_interventional notin {'Epidemiological study', 'N/A','Observational study','Other'}
observational_model notin {'Case Control','Case Crossover','Case Only','Cohort','Ecologic or Community','N/A','Natural History','Other'}
ctgov_purpose notin {'Basic Science','Diagnostic','Health Services Research', 'Other','Prevention','Screening','Supportive Care','Treatment'}
drks_purpose notin {'Basic research/physiological study','Diagnostic','Health care system', 'Other','Prevention','Prognosis','Screening','Supportive care','Treatment'}

--Interaction constraints
study_type_non_interventional in {'Epidemiological study', 'Observational study', 'Other'} & ctgov_purpose in {'Treatment'}
study_type_non_interventional in {'Epidemiological study', 'Observational study', 'Other'} & drks_purpose in {'Treatment'}
ctgov_purpose in {'Basic Science', 'Diagnostic', 'Health Services Research', 'Other', 'Prevention', 'Supportive Care', 'Treatment'} & drks_purpose in {'Screening'}
ctgov_purpose in {'Basic Science', 'Diagnostic', 'Health Services Research', 'Other', 'Prevention', 'Supportive Care', 'Screening'} & drks_purpose in {'Treatment'}
ctgov_purpose in {'Basic Science', 'Health Services Research', 'Other', 'Prevention', 'Supportive Care', 'Screening', 'Treatment'} & drks_purpose in {'Diagnostic'}
ctgov_purpose in {'Diagnostic', 'Health Services Research', 'Other', 'Prevention', 'Supportive Care', 'Screening', 'Treatment'} & drks_purpose in {'Basic research/physiological study'}
ctgov_purpose in {'Basic Science', 'Diagnostic', 'Health Services Research', 'Other', 'Prevention', 'Screening', 'Treatment'} & drks_purpose in {'Supportive care'}
ctgov_purpose in {'Basic Science', 'Diagnostic', 'Other', 'Prevention', 'Supportive Care', 'Screening', 'Treatment'} & drks_purpose in {'Health care system'}
ctgov_purpose in {'Basic Science', 'Diagnostic', 'Health Services Research', 'Prevention', 'Supportive Care', 'Screening', 'Treatment'} & drks_purpose in {'Other'}
ctgov_purpose in {'Basic Science', 'Diagnostic', 'Health Services Research', 'Prevention', 'Supportive Care', 'Screening', 'Treatment'} & drks_purpose in {'Prognosis'}
ctgov_purpose in {'Basic Science', 'Diagnostic', 'Health Services Research', 'Other', 'Supportive Care', 'Screening', 'Treatment'} & drks_purpose in {'Prevention'}
ctgov_study_type in {'Interventional'} & study_type_non_interventional in {'Epidemiological study', 'Observational study', 'Other'}
ctgov_study_type in {'Observational', 'Observational [Patient Registry]'} & study_type_non_interventional in {'N/A'}
drks_study_type in {'Interventional'} & study_type_non_interventional in {'Epidemiological study', 'Observational study', 'Other'}
drks_study_type in {'Non-interventional'} & study_type_non_interventional in {'N/A'}
study_type_non_interventional in {'N/A'} & observational_model in {'Case Control', 'Case Crossover', 'Case Only', 'Cohort', 'Ecologic or Community', 'Natural History', 'Other'}
study_type_non_interventional in {'Epidemiological study', 'Observational study', 'Other'} & observational_model in {'N/A'}