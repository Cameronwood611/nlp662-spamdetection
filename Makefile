init:
	pdm install
	npm install

start-backend:
	pdm run bash -c "export FLASK_ENV=development; pdm run python ./www/public/app.py"

start-frontend:
	npm run start