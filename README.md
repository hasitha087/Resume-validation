# Resume-validation
This will identify whether a document is CV or non-CV.

- Drag and drop your training sample into training_data folder(use only PDF format only for the training). Put CVs in CV folder and non-CVs in non-CV folder.
- Once you train the model(run classifier_model.py), model files(pickle) save under model folder(there will be two pickle files. One for classification model and other is for save the vectors).
- Once you train the model you can run the prediction on your documents. Output saves in output folder and seperate the documents as CV,non-CV and Exception(Files which unable to handle).
- This model can handle following file types.
    - doc, docx, odt
    - pdf
    - pptx
    - odp
    - zip files(zip files will extract into zipextract folder)
