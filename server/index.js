const express = require('express');
const morgan = require('morgan');
const bodyParser = require('body-parser')
let app = express();

app.use(morgan('dev'))

app.use(express.static(path.join(__dirname, '../public')));

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.use('/api', require('./routes'))

app.use((req, res, next) => {
  console.log(`${new Date()} - ${req.method} request for ${req.url}`);
  next()
});
