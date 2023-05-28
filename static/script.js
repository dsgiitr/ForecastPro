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

  // Function to fetch test and prediction data from Flask server
  function fetchData() {
    fetch('/data')
      .then(response => response.json())
      .then(data => {
        // Extract test and prediction data from the response
        var { test, predictions } = data;
  
        test = test.map(element => element[0])
        predictions = predictions.map(element => element[0])
  
        addChart(test, predictions)
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }
  
  
  function addChart(test, predictions){
    // Chart data
    var data = {
      labels : [...Array(test.length).keys()],
      datasets : [
          {
            data : test,
            label : "Test",
            borderColor : "#3cba9f",
          },
          {
            data : predictions,
            label : "Predictions",
            borderColor : "#FF5D53",
          }
        ]
    };
  
    // Chart options
    var options = {
      responsive: true,
      // scales: {
      //   y: {
      //     beginAtZero: true
      //   }
      // }
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
            y: {
              max: 'original',
              min: 'original'
            }
          }
        }
      }
  
    };
  
    // Create the chart
    new Chart(document.getElementById("chart"), {
      type : 'line',
      data : data,
      options : options
    });
  
  }
  
  window.addEventListener('load', fetchData);
  
  var ctx = document.getElementById('chart').getContext('2d');
  
  } catch(err) {
    console.error(err)
    
  }