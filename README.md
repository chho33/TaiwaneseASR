# taiwanese-asr

## Warning
This repository is only for demo purpose. Since opening microphone on browsers need https, this project support only running locally so far.

## Pre-requests
- Download the model: https://drive.google.com/file/d/1VeSKm1F1t4b1eD73UfRBoiGY_Zhv0-9B/view?usp=sharing
- Decompress the model and move it into the right directory: `tar zxf run.tar.gz; mv run ./backend/;`
- install npm: https://www.npmjs.com/get-npm
- install yarn: https://classic.yarnpkg.com/en/docs/install
- `npm install -g @vue/cli`
- `yarn install`
- `vue add axios`

## Frontend
```
cd src;
yarn serve
```

## Backend
```
cd backend;
python server.py
```
