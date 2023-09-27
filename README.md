# Singapore Rental Market Insights

Welcome to the Singapore Rental Market Insights project. This project aims to provide valuable insights into the rental property market in Singapore. Whether you're a newcomer to the city or a local resident looking to rent, these insights can help you make informed decisions.

I have built a Tableau Data Story, feel free to explore [Tableau Data Story](https://public.tableau.com/views/RentalPropertyDashboard/DataStory?:language=en-US&:display_count=n&:origin=viz_share_link).

## Project Overview

This project consists of four main phases:

1. **Web Scraping**: In the `Web_Scraping.ipynb` notebook, we extract data from a fixed website and save it into `property_listing.csv`.

2. **Data Cleaning**: The `Data_Cleaning.ipynb` notebook reads the `property_listing.csv` file, cleans, and processes the data, saving it into `Rental_data.csv`.

3. **Feature Engineering**: The `Feature_Engineering.ipynb` notebook reads the `Rental_data.csv` file, extracts new features, and saves the final dataset as `Rental_data_final.csv`.

4. **Data Analysis**: The `Data_Analysis.ipynb` notebook reads the `Rental_data_final.csv` file and proceeds to Exploratory Data Analysis.

## Getting Started

To explore the project and its findings, follow these steps:

1. Clone the repository to your local machine:

2. Open and run the Jupyter notebooks to view and analyze the data:
- `Web_Scraping.ipynb`
- `Data_Cleaning.ipynb`
- `Feature_Engineering.ipynb`
- `Data_Analysis.ipynb`

4. Explore the project's visualizations, insights, and findings.

## Project Structure

- `Web_Scraping.ipynb`: Web scraping script to gather rental property data.
- `Data_Cleaning.ipynb`: Data cleaning and preprocessing script.
- `Feature_Engineering.ipynb`: Feature extraction and final dataset creation.
- `Data_Analysis.ipynb`: EDA process.
- `property_listing.csv`: Raw data extracted from the website.
- `Rental_data.csv`: Cleaned and processed rental data.
- `Rental_data_final.csv`: Final dataset with engineered features.

## Data Sources

- Property data was collected from [99.co](https://www.99.co/).

## Visualizations and Insights

- The project includes visualizations and insights into rental prices, regional variations, property types, room types, and more. Refer to the Jupyter notebooks for detailed analysis.

## Future Improvements and Additions

While the current version of the project provides valuable insights into the Singapore rental market, there are several areas for improvement and additional features that I plan to work on in the future:

1. **Additional Data Sources**: I intend to incorporate data from the PropertyGuru website to enhance the dataset and provide a more comprehensive view of the rental market in Singapore.

2. **Data Processing Enhancement**: The data processing pipeline can be further refined to address issues with missing room type and room size information. I will continue to refine the logic rules used to assign values and correct any inaccuracies.

3. **Optimizing Geocoding**: Geocoding addresses can be time-consuming. I'm exploring alternative geocoding solutions to reduce the processing time and improve accuracy. If you have suggestions or know of better tools, please feel free to reach out.

4. **Subjectivity in Analysis**: The analysis presented in this project is based on my interpretation and understanding of the data. I welcome feedback and differing opinions from others in the community. If you have alternative insights or perspectives, don't hesitate to contact me.

5. **Ongoing Optimization**: There are always opportunities for optimization and enhancement in this project. If you have ideas or want to contribute to further improve the project, please don't hesitate to get involved.

Your feedback and contributions are highly valued, and I look forward to making this project even more valuable and insightful in the future.


## Contact

For any questions or inquiries about this project, please contact lowminghaw@gmail.com.

