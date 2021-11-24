# rcic

This API can be accessed using https://rcic.herokuapp.com/docs

To make a prediction, click "Try it out" in the Swagger GUI and click "Add string item" to add a file input. Repeat this once for each image. This model currently only predicts siRNA treatment from six-channel images, split into one channel per-PNG file. Once all six files have been chosen, click "Execute". The predicted siRNA treatment value will appear in the Responses section of the page.

This repo contains all necessary files to re-create the deployment of this API. To do so, connect your repository to your Heroku project and deploy manually or automatically upon commit.
