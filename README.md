
# Social Network: Rechype

Developed a social networking application centered around the sharing of culinary ideas and recipes. Focusing on the implementation of non-relational databases.

Project completed for the “Large-Scale and Multi-Structured Databases” exam.



## Short Description

Rechype is a social networking application centered around the sharing of culinary ideas and recipes.
We employed various non-relational databases, such as MongoDB, Neo4j, and HaloDB.

To extract valuable insights from the data, we implemented specific aggregation pipelines.

Furthermore, we conducted an assessment of various indexing methods to enhance the read performance of the database.

Finally we implemented some routines to maintain cross-database consistency since this is a critical aspect of the application.

Registered users have the option to either create new recipes or select from a constantly updated collection provided by the system, which is curated by fellow users. The system offers recipes sourced from multiple APIs, including cocktailDB, Spoonacular API, and PunkAPI. The first API contains cocktail recipes, the second primarily focuses on food recipes, and the last one offers a variety of beer-related recipes.

Users can access a comprehensive list of ingredients, extracted from the Spoonacular API, which allows them to curate their own virtual fridge and invent new recipes and drinks. The user interface offers a feature to assemble meals, which are combinations of different recipes and beverages. These meals can be labeled with various attributes such as name and type (e.g., breakfast, brunch).
