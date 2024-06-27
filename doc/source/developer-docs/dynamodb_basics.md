# DynamoDB Basics in AWS
This is a document that highlights key aspects of DynamoDB in AWS and how they are relevant to the Libera SDC. To
understand the Libera data model, it is important to understand the basics of DynamoDB and how it works. This document
provides an overview of the DynamoDB basics and how they are used in the Libera databases and is not
intended to be a comprehensive guide to DynamoDB. For more information on DynamoDB, see the 
[AWS DynamoDB documentation](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html).

See the [Libera Data Models and Database Schema for DynamoDB in AWS](database_data_models.md) for more information on the
specific data models used in the Libera databases.

## DyanmoDB Formal Description
DynamoDB is a serverless NoSQL database service provided by AWS. It is a key-value and document database. It is a fully 
managed database with built-in security, backup and restore, and in-memory caching for internet-scale 
applications. DynamoDB uses tables, items, and attributes as the core components that you work with. A table is a 
collection of items, and each item is a collection of attributes. DynamoDB uses primary keys to uniquely identify each 
item in a table. 

**Important Note:** The key-value nature of DynamoDB specifically uses only strings for keys and has a limited set of
data types available for values. Libera SDC at this time only utilizes string and numeric data types in the
DynamoDB tables.

### Costs
At its simplest, DyanmoDB usage is billed based on the number and size of read and write operations. The design of the 
database schema can have a significant impact on the cost of using DynamoDB. The Libera databases are designed to take
advantage of the DynamoDB pricing model to minimize costs with small read operations by using the concept of vertical 
scaling. See the vertical scaling section later in this documentation.

### Event Driven Architecture
DynamoDB is designed for use in an event-driven architecture. This means that you can trigger events based on changes
to the database. This is useful for triggering Lambda functions, SNS notifications, or other AWS services based on changes
to the database. DynamoDB Streams captures a time-ordered sequence of item-level modifications in any DynamoDB table 
and can pass these changes to a Lambda function for processing.

Libera specifically makes use of the DynamoDB Streams feature to trigger Lambda functions when changes are made to the
database. This is used to coordinate downstream processing of different processing steps using Lambda and Step Functions. 
The data models for the Libera databases are designed to take advantage of this feature while stil using the vertical 
scaling pattern to minimize costs.

### DynamoDB Primary Keys
To access specific items DynamoDB supports two types of primary keys to **uniquely** access items in a table:
- Partition key: A simple primary key, composed of one attribute known as the partition key.
- Partition key and sort key: A composite primary key, composed of two attributes. The first attribute is the partition key,
and the second attribute is the sort key.

Libera uses the partition key and sort key access pattern.

### Reading Data Well
DynamoDB allows for three different methods of reading data:
- GetItem: Retrieves a single item from a table.
- Query: Retrieves all items that have the same partition key.
- Scan: Retrieves all items in a table.

Note that the Query and Get operations are the most efficient way to access data in DynamoDB as
it allows you to access data based on specific keys. The Scan operation is the least efficient way to access data
as it reads all items in the table and can be very expensive. Avoid using the Scan operation whenever possible as the
tables get large.

### DynamoDB Secondary Indexes
DynamoDB supports secondary indexes to allow you to query the data in a table using an alternate key. This is used by
Libera to define specific "Use Cases" that allow for querying the data in different ways from the standard primary key.

As a specific example, the Libera metadata database functions primarily as a record of each file that has been processed
where the unique primary key in the database is the filename and one of the attributes is the date for which the data
in the file was collected. If we wanted to access all files that relate to a specific date just using the primary key, 
we would need to scan the whole database then search on the specific date attribute. This would be very inefficient and 
potentially expensive. Instead, we can define a secondary index on the date attribute and then query the database using
the date attribute as the key. This allows us to use a Query command to access the data we need and get back only the
items in the database that have the specific date attribute of the date we are interested in. Further details on the 
specific secondary indexes used in the Libera databases are provided in the following sections. Further details on
DynamoDB secondary indexes can be found in the [AWS Secondary Index documentation](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/SecondaryIndexes.html).

### Vertical Partitioning
Vertical partitioning is a database design pattern that involves effectively splitting a table into smaller tables 
based on the attributes that are most frequently accessed together. This can be used to optimize the performance of
queries and reduce the cost of accessing data in the database. In the Libera databases, we use vertical partitioning
with the sort key to allow for efficient access to the data in the database. This is done by defining the sort key
as a string that is a concatenation of the attributes that are most frequently accessed together. This allows us to
use the Query operation to access the data we need and get back only the items in the database that have the specific
sort key attribute of the sort key we are interested in. This is a key design pattern used in the Libera databases to
optimize the performance and cost of accessing data in the database. Further details on vertical partitioning can be
found in the [AWS Vertical Partitioning blog post](https://aws.amazon.com/blogs/database/use-vertical-partitioning-to-scale-data-efficiently-in-amazon-dynamodb/).

