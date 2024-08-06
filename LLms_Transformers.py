from transformers import pipeline
import pandas as pd

# 1. POSITIVE / NEGATIVE COMMENTS
# Text for positive and negative comments
text_positive = """Customer: The implementation of the new feature is excellent. Great job!"""
text_negative = """There are several bugs in the new release, and the system crashes frequently."""

# Create a classifier object with the pipeline method
classifier = pipeline('sentiment-analysis', 'distilbert/distilbert-base-uncased-finetuned-sst-2-english')
res = classifier(text_negative)
res_1 = classifier(text_positive)
print(res, res_1)

# [{'label': 'NEGATIVE', 'score': 0.999735414981842}] [{'label': 'POSITIVE', 'score': 0.9998588562011719}]

# 2. ENTITY RECOGNITION FOR DOCUMENTATION
# Text for entity recognition analysis
text = """ Dr. Emily Phillips from Healthcare Corp is working with developers from Very-cool-corporation Inc. to implement a new feature for electronic health records (EHR) module at their facilities in Berlin. """

# Create a ner_tagger object with the pipeline method
ner_tagger = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Analyze the text for entity recognition
outputs = ner_tagger(text)

# Display results in a DataFrame for convenient visualization and structuring of documentation
df = pd.DataFrame(outputs)
print(df)

# Structuring information for the project
structured_info = {
    'Person': df[df['entity_group'] == 'PER']['word'].tolist(),
    'Organization': df[df['entity_group'] == 'ORG']['word'].tolist(),
    'Location': df[df['entity_group'] == 'LOC']['word'].tolist()
}

# Display structured information
print("\nStructured Information:")
for key, value in structured_info.items():
    print(f"{key}: {', '.join(value)}")

# Example of auto-filling documentation
def generate_documentation(info):
    doc_template = f"""
    Project Documentation
    Participants:
    - Persons: {', '.join(info['Person'])}
    - Organizations: {', '.join(info['Organization'])}
    - Locations: {', '.join(info['Location'])}
    Description:
    {text}
    Tasks:
    - Collaborate with {info['Organization'][0]} to implement a new feature for the EHR module.
    - Ensure that the integration with the healthcare facilities in {info['Location'][0]} is seamless.
    """
    return doc_template

# Display generated documentation
documentation = generate_documentation(structured_info)
print("\nGenerated Documentation:")
print(documentation)

# Some model weights messages...

# 3. SUMMARIZATION: GENERAL EXAMPLE
# Example of a long task description
text = """ The main goal of this project is to develop and implement a new feature for the electronic health records (EHR) system used by healthcare professionals at Healthcare Corp. The new feature aims to improve the efficiency of patient data entry by providing an intuitive user interface and integrating automated data verification processes. This will include several sub-tasks such as designing the user interface, developing back-end services for data validation, and conducting extensive testing to ensure the system's reliability and accuracy. Additionally, the project will involve training healthcare staff to effectively use the new feature and gathering feedback for further improvements. The timeline for the project spans six months, with key milestones set at the end of each month to track progress and address any arising issues. """

# Create a summarizer object with the pipeline method
summarizer = pipeline("summarization", model='facebook/bart-large-cnn')

# Generate a summary
outputs = summarizer(text, max_length=60, min_length=30, clean_up_tokenization_spaces=True, do_sample=False)

# Display summary
print("Summary of the project:\n")
print(outputs[0]['summary_text'])

# Summary of the project: The project aims to develop and implement a new feature for the electronic health records (EHR) system. The new feature aims to improve the efficiency of patient data entry by providing an intuitive user interface.

# 3.1 SUMMARIZATION: REQUIREMENTS SUMMARY AND FULL VERSION
def generate_requirements_summary(requirement_text):
    summarizer = pipeline("summarization", model='facebook/bart-large-cnn')
    summary_output = summarizer(requirement_text, max_length=60, min_length=30, clean_up_tokenization_spaces=True, do_sample=False)
    return summary_output[0]['summary_text']

# Example of a long requirement
requirement_text = """ The Part-11 regulation by the FDA requires comprehensive controls for electronic records and electronic signatures to ensure data integrity and security. This includes implementing access controls to limit system access to authorized individuals, ensuring that electronic records can be accurately and readily retrieved throughout the retention period, and providing a secure environment for the use of electronic signatures. Part-11 also mandates that systems ensure the authenticity, integrity, and, when appropriate, the confidentiality of electronic records, and to ensure that the signer cannot readily repudiate the signed record as not genuine. Additionally, the validation of system functionality and the ability to generate accurate and complete copies of records in both human-readable and electronic form are required. Regular system audits and documentation of operational and security controls are critical to compliance. """

# Get the summary
summary = generate_requirements_summary(requirement_text)

# Example of auto-filling documentation
def generate_compliance_documentation(summary):
    doc_template = f"""
    Compliance Summary for Part-11 Regulation
    Summary:
    {summary}
    Detailed Requirements:
    {requirement_text}
    """
    return doc_template

# Generate and display documentation
compliance_doc = generate_compliance_documentation(summary)
print("\nGenerated Documentation:")
print(compliance_doc)

# Generated Documentation:
# Summary: The Part-11 regulation by the FDA requires comprehensive controls for electronic records and electronic signatures...

# 4. ANSWERING QUESTIONS
# Example text for functional description
text = """ The latest version of the Very-cool-corporation product lifecycle management software includes an updated compliance module that ensures all documentation meets FDA and Part-11 regulations. Users can now automate the generation of audit reports, track changes in real-time, and maintain a complete version history. The new dashboard interface provides intuitive navigation and comprehensive data visualization tools. Integration with AWS cloud services allows seamless scalability and data security. Additionally, the system supports multi-user collaboration, enabling teams to work together efficiently across different locations. """

# Create a question_answerer object with the pipeline method
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

# Questions for the text
question = "What does the compliance module ensure?"
question_1 = "What features does the dashboard interface provide?"
question_2 = "How does the system support multi-user collaboration?"

# Get answers to the questions
outputs = question_answerer(question=question, context=text)
outputs_1 = question_answerer(question=question_1, context=text)
outputs_2 = question_answerer(question=question_2, context=text)

# Display answers in a DataFrame for convenient visualization
print(pd.DataFrame([outputs]))
print(pd.DataFrame([outputs_1]))
print(pd.DataFrame([outputs_2]))

# Answers displayed...

# 4.1 QUESTION: WHAT IS THE THEME OF Very-cool-corporation WEBINAR (LINKEDIN)
# Example of a long task description from Very-cool-corporation LinkedIn page
text = """ This webinar will teach participants how to achieve seamless documentation and traceability for automated tests in compliance with IEC 62304 standards. FDA-compliant SBOM is distinctly different from a standard SBOM. There are SBOM generators and tools available that can assist in the creation and management of standard SBOMs. However, there is still a large amount of manual intervention and configuration that falls on the development and compliance teams who use them to create an FDA-compliant SBOM. Join us for our most popular webinar on Wednesday, May 29th to learn how to transform Jira into 62304 compliance. In this webinar, we'll cover the following: ✅ Clarify Jira Gaps with regard to 62304: Learn Jira’s critical gaps in meeting IEC 62304 and the challenges posed. ✅ Transform Jira into 62304 compliance: Learn effective strategies for integrating Jira into regulated development workflows, with a focus on customization and configuration to align with IEC 62304 requirements. ✅ Use Jira for Device/SaMD development: Review a live, 62304-compliant Jira instance automatically configured by Very-cool-corporation. Join us on May 15 for our next webinar. Learn how to empower your SDLC process with tools to help build a more efficient and collaborative partnership between R&D and Quality teams. Join us for our next webinar on April 18th where we'll provide an in-depth understanding of how GitHub, GitLab, and Bitbucket can be leveraged to meet the stringent requirements of the IEC 62304. Participants will learn practical strategies for tracing unit and automated tests from Git repositories to test cases and requirements, managing design controls for multiple product versions, and implementing common engineering controls in GitHub/GitLab to ensure Total Product Life Cycle (TPLC) control. """

# Split text into logical parts
context_1 = """ This webinar will teach participants how to achieve seamless documentation and traceability for automated tests in compliance with IEC 62304 standards. """

# Use the model for text summarization
summarizer = pipeline("summarization", model='facebook/bart-large-cnn')

# Generate summaries for each context
summary_1 = summarizer(context_1, max_length=60, min_length=30, clean_up_tokenization_spaces=True)[0]['summary_text']
print("Summary 1:", summary_1)

# Create a question_answerer object with the pipeline method
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

# Questions for the summarized contexts (refined)
question_1 = "What is the theme of the webinar?"

# Get answers to the questions from the summaries
outputs_1 = question_answerer(question=question_1, context=summary_1)

# Display answers in a DataFrame for convenient visualization
print(pd.DataFrame([outputs_1]))
