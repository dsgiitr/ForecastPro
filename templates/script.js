function generateChart(data) {
    var ctx = document.getElementById('chart').getContext('2d');
    
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({ length: data.length }, (_, i) => i + 1), // Assuming labels are numeric 1, 2, 3...
        datasets: [{
          label: 'Data from pickle file',
          data: data,
          backgroundColor: 'rgba(0, 123, 255, 0.2)',
          borderColor: 'rgba(0, 123, 255, 1)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: 'X-axis'
            }
          },
          y: {
            display: true,
            title: {
              display: true,
              text: 'Y-axis'
            }
          }
        }
      }
    });
  }
  