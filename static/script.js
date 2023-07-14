try{

// JavaScript code to handle data source selection
document.getElementById('data-source').addEventListener('change', function() {
  var selectedOption = this.value;
  var stockListDiv = document.getElementById('stock-list');
  var customDataDiv = document.getElementById('custom-data');

  if (selectedOption === 'yahoo') {
    stockListDiv.style.display = 'block';
    customDataDiv.style.display = 'none';
  } else if (selectedOption === 'custom') {
    stockListDiv.style.display = 'none';
    customDataDiv.style.display = 'block';
  }
});

document.getElementById("train-model-btn").addEventListener("click", function(event) {
  event.preventDefault();
  console.log("hiiii")

  var form = document.getElementById("train-model-form");
  var formData = new FormData(form);

  fetch("/train-model", {
    method: "POST",
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    console.log(data);
    addChart("new", data.test, data.predictions, data.forecast)
  })
  .catch(error => {
    console.error(error);
  });
});

// Function to fetch test and prediction data from Flask server
async function fetchData (predfile, testfile, func) {

  await fetch(
    '/data', {
      method: "post",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        "predfile": predfile,
        "testfile": testfile
      })
    })
    .then(response => response.json())
    .then(data => {

      var { test, predictions } = data;

      func(test, predictions)

    })
    .catch(error => {
      console.error('Error:', error);
    });
}


function addChart(chartId, test, predictions, forecast=[]){
  var data = {
    labels : [...Array(test.length).keys()],
    datasets : [
        {
          data : test,
          label : "Test",
          borderColor : "#3CBA9F",
          showLine: true,
        },
        {
          data : predictions,
          label : "Predictions",
          borderColor : "#FF5D53",
          showLine: true,

        },
        {
          data : forecast,
          label : "Forecast",
          borderColor : "#1585B3",
          showLine: true,

        },
      ]
  };

  var options = {
    plugins: {
      zoom: {
        pan: {
          enabled: true
        },
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true
          },
          mode: 'xy',
        },
        limits : {
          x: {
            max: 'original',
            min: 'original'
          },

          y: {
            max: 'original',
            min: 'original'
          }
        }
      }
    }

  };

  // Create the chart
  new Chart(document.getElementById(chartId), {
    type : 'scatter',
    data : data,
    options : options
  });

}

window.addEventListener('load', ()=>{
  fetchData('static/past_preds.json', 'static/past_test.json', (test, predictions)=>{
    addChart("past", test, predictions)
  })
});



// var ctx = document.getElementById("").getContext('2d');

} catch(err) {
  console.error(err)

}