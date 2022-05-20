#sigma

-- Contracts
@study_type_non_interventional:string
@ctgov_purpose:string
@drks_purpose:string
@drks_study_type:string
@ctgov_study_type:string
@observational_model:string

--Domain constraints

study_type_non_interventional notin {'N/A','Observational study','Epidemiological study','Other'}
ctgov_study_type notin {'Observational [Patient Registry]','Observational','Interventional'}
drks_purpose notin {'Prevention','Screening','Health care system','Prognosis','Treatment','Basic research/physiological study','Supportive care','Diagnostic','Other'}
observational_model notin {'N/A','Case Control','Cohort','Ecologic or Community','Case Crossover','Case Only','Natural History','Other'}
ctgov_purpose notin {'Prevention','Screening','Treatment','Basic Science','Diagnostic','Supportive Care','Health Services Research','Other'}
drks_study_type notin {'Interventional','Non-interventional'}

--Interaction constraints

drks_purpose in {'Treatment'} & observational_model in {'Case Control','Cohort','Ecologic or Community','Case Crossover','Case Only','Natural History','Other'}
ctgov_study_type in {'Observational [Patient Registry]','Observational'} & drks_purpose in {'Treatment'}
observational_model in {'N/A'} & study_type_non_interventional in {'Observational study','Epidemiological study','Other'}
study_type_non_interventional in {'Observational study','Epidemiological study','Other'} & ctgov_purpose in {'Treatment'}
ctgov_purpose in {'Prevention','Screening','Treatment','Basic Science','Diagnostic','Supportive Care','Other'} & drks_purpose in {'Health care system'}
drks_purpose in {'Prognosis'} & ctgov_purpose in {'Prevention','Screening','Treatment','Basic Science','Diagnostic','Supportive Care','Health Services Research'}
ctgov_study_type in {'Observational [Patient Registry]','Observational'} & ctgov_purpose in {'Treatment'}
drks_purpose in {'Screening'} & ctgov_purpose in {'Prevention','Treatment','Basic Science','Diagnostic','Supportive Care','Health Services Research','Other'}
drks_purpose in {'Treatment'} & ctgov_purpose in {'Prevention','Screening','Basic Science','Diagnostic','Supportive Care','Health Services Research','Other'}
ctgov_purpose in {'Prevention','Screening','Treatment','Basic Science','Supportive Care','Health Services Research','Other'} & drks_purpose in {'Diagnostic'}
drks_purpose in {'Other'} & ctgov_purpose in {'Prevention','Screening','Treatment','Basic Science','Diagnostic','Supportive Care','Health Services Research'}
drks_purpose in {'Prevention'} & ctgov_purpose in {'Screening','Treatment','Basic Science','Diagnostic','Supportive Care','Health Services Research','Other'}
study_type_non_interventional in {'Observational study','Epidemiological study','Other'} & drks_study_type in {'Interventional'}
study_type_non_interventional in {'N/A'} & drks_study_type in {'Non-interventional'}
study_type_non_interventional in {'N/A'} & observational_model in {'Case Control','Cohort','Ecologic or Community','Case Crossover','Case Only','Natural History','Other'}
observational_model in {'N/A'} & drks_study_type in {'Non-interventional'}
ctgov_study_type in {'Interventional'} & observational_model in {'Case Control','Cohort','Ecologic or Community','Case Crossover','Case Only','Natural History','Other'}
study_type_non_interventional in {'N/A'} & ctgov_study_type in {'Observational [Patient Registry]','Observational'}
ctgov_purpose in {'Treatment'} & drks_study_type in {'Non-interventional'}
ctgov_purpose in {'Prevention','Screening','Treatment','Basic Science','Diagnostic','Health Services Research','Other'} & drks_purpose in {'Supportive care'}
ctgov_study_type in {'Observational [Patient Registry]','Observational'} & drks_study_type in {'Interventional'}
ctgov_purpose in {'Treatment'} & observational_model in {'Case Control','Cohort','Ecologic or Community','Case Crossover','Case Only','Natural History','Other'}
ctgov_study_type in {'Observational [Patient Registry]','Observational'} & observational_model in {'N/A'}
study_type_non_interventional in {'Observational study','Epidemiological study','Other'} & drks_purpose in {'Treatment'}
drks_study_type in {'Non-interventional'} & ctgov_study_type in {'Interventional'}
drks_study_type in {'Interventional'} & observational_model in {'Case Control','Cohort','Ecologic or Community','Case Crossover','Case Only','Natural History','Other'}
ctgov_purpose in {'Prevention','Screening','Treatment','Diagnostic','Supportive Care','Health Services Research','Other'} & drks_purpose in {'Basic research/physiological study'}
drks_purpose in {'Treatment'} & drks_study_type in {'Non-interventional'}
study_type_non_interventional in {'Observational study','Epidemiological study','Other'} & ctgov_study_type in {'Interventional'}

#assertions


#fd

