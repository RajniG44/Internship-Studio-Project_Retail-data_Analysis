CREATE database RetailSalesData;
Use RetailSalesData;
DROP DATABASE retailsalesdata;
USE retailsalesdata; 
DROP TABLE Sales_Data_Transactions;
CREATE DATABASE RetailSalesData;
USE RetailSalesData;

CREATE TABLE Sales_Data_Transactions (
customer_id VARCHAR(255),
trans_date DATETIME,
tran_amount INT);

CREATE TABLE Sales_Data_Response(
customer_id VARCHAR(255) PRIMARY KEY ,
trans_date DATETIME,
tran_amount INT);

# secure-file-priv="C:/ProgramData/MySQL/MySQL Server 8.0/Uploads"
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Retail_Data_Transactions.csv'
INTO TABLE Sales_Data_Transactions
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

EXPLAIN SELECT * FROM Sales_Data_Transactions WHERE CUSTOMER_ID= 'CS5295';
Select * From Sales_Data_Transactions LIMIT 10;

CREATE INDEX idx_id ON Sales_Data_Transactions(CUSTOMER_ID);
 EXPLAIN SELECT * FROM Sales_Data_Transactions WHERE CUSTOMER_ID= 'CS5295';
 

C:\Users\HP\Downloads\Internship Studio Project\retail data project.sql does not contain schema/table information
Dump file not found: File C:\Users\HP\OneDrive\Documents\dumps\export.sql doesn't exist
Dump file not found: File C:\Users\HP\OneDrive\Documents\dumps\export.sql doesn't exist
14:08:51 Restoring C:\Users\HP\Downloads\Internship Studio Project\retail data project.sql
Running: mysql.exe --defaults-file="C:\Users\HP\AppData\Local\Temp\tmptb3vml_c.cnf"  --protocol=tcp --host=localhost --user=root --port=3306 --default-character-set=utf8 --comments --database=retailsalesdata  < "C:\\Users\\HP\\Downloads\\Internship Studio Project\\retail data project.sql"
ERROR 1007 (HY000) at line 1: Can't create database 'retailsalesdata'; database exists

DROP DATABASE IF EXISTS retailsalesdata;
CREATE DATABASE retailsalesdata;

