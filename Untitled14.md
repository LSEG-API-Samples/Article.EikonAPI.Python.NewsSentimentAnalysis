
# Introduction to News Sentiment Analysis with Eikon Data APIs - a Python example

This article will demonstrate how we can conduct a simple sentiment analysis of news delivered via our new [Eikon Data APIs](https://developers.thomsonreuters.com/eikon-data-apis). Natural Language Processing (NLP) is a big area of interest for those looking to gain insight and new sources of value from the vast quantities of unstructured data out there. The area is quite complex and there are many resources online that can help you familiarise yourself with this very interesting area. There are also many different packages that can help you as well as many different approaches to this problem. Whilst these are beyond the scope of this article - I will go through a simple implementation which will give you a swift enough introduction and practical codebase for further exploration and learning.

Pre-requisites: 

Thomson Reuters Eikon with access to new [Eikon Data APIs](https://developers.thomsonreuters.com/eikon-data-apis)

Python 2.x/3.x

Required Python Packages: eikon, pandas, numpy, beautifulsoup, textblob, datetime 

Required corpora download: python -m textblob.download_corpora (this is required by the sentiment engine to generate sentiment)

### Introduction

NLP is a field which enables computers to understand human language (whether by speech or by text). This is quite a big area of research and a little enquiry on your part will furnish you with the complexities of this problem set. Here we will be focussing on one application of this called Sentiment Analysis. In our case we will be taking news articles(unstructured text) for a particular company, IBM, and we will attempt to grade this news to see how postive, negative or neutral it is. We will then try to see if this news has had an impact on the shareprice of IBM. 

To do this really well is a non-trivial task, and most universtities and financial companies will have departments and teams looking at this. We ourselves provide machine readable news products with News Analytics (such as sentiment) over our Elektron platform in realtime at very low latency - these products are essentially consumed by algorithmic applications as opposed to humans. 

We will try to do a similar thing as simply as possible to illustrate the key elements - our task is significantly eased by not having to do this in a low latency environment. We will be abstracting most of the complexities to do with the mechanics of actually analysing the text to various packages. You can then easily replace the modules such as the sentiment engine etc to improve your results as your understanding increases.  

So lets get started. First lets load the packages that we will need to use and set our app_id. 


```python
import eikon as ek
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from textblob import TextBlob
import datetime
from datetime import time
import warnings
warnings.filterwarnings("ignore")
ek.set_app_id('D8AEEB197B8AAF7FCEBF9')
```

There are two API calls for news:

**get_news_headlines** : returns a list of news headlines satisfying a query

**get_news_story** : returns a HTML representation of the full news article

We will need to use both - thankfully they are really straightforward to use. We will need to use **get_news_headlines** API call to request a list of headlines. The first parameter for this call is a query. You dont really need to know this query language as you can generate it using the **News Monitor App** (type **NEWS** into Eikon search bar) in **Eikon**. 

You can see here I have just typed in 2 search terms, **IBM**, for the company, and, **English**, for the language I am interested in (in our example we will only be able to analyse English language text - though there are corpora, packages, methods you can employ to target other languages - though these are beyond the scope of this article). You can of course use any search terms you wish.

![News App 1](Article 14 One.jpg)

After you have typed in what you want to search for - we can simply click in the search box and this will then generate the query text which we can then copy and paste into the API call below. Its easy for us to change logical operations such as **AND** to **OR**, **NOT**  to suit our query. 

![News App 2](Article 14 Two.png)

So the line of code below gets us 100 news headlines for **IBM** in english prior to 4th Dec 2017, and stores them in a dataframe, df for us.


```python
df = ek.get_news_headlines('R:IBM.N AND Language:LEN', date_to = "2017-12-04", count=100)
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>versionCreated</th>
      <th>text</th>
      <th>storyId</th>
      <th>sourceCode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-12-01 23:11:47.374</th>
      <td>2017-12-01 23:11:47.374</td>
      <td>Reuters Insider - FM Final Trade: HAL, TWTR &amp; ...</td>
      <td>urn:newsml:reuters.com:20171201:nRTV8KBb1N:1</td>
      <td>NS:CNBC</td>
    </tr>
    <tr>
      <th>2017-12-01 19:19:20.279</th>
      <td>2017-12-01 19:19:20.279</td>
      <td>IBM ST: the upside prevails as long as 150.2 i...</td>
      <td>urn:newsml:reuters.com:20171201:nGUR2R6xQ:1</td>
      <td>NS:GURU</td>
    </tr>
    <tr>
      <th>2017-12-01 18:12:41.143</th>
      <td>2017-12-01 18:12:41.143</td>
      <td>INTERNATIONAL BUSINESS MACHINES CORP SEC Filin...</td>
      <td>urn:newsml:reuters.com:20171201:nEOL6JJfRT:1</td>
      <td>NS:EDG</td>
    </tr>
    <tr>
      <th>2017-12-01 18:12:41.019</th>
      <td>2017-12-01 18:12:41.019</td>
      <td>INTERNATIONAL BUSINESS MACHINES CORP SEC Filin...</td>
      <td>urn:newsml:reuters.com:20171201:nEOL3YHcVY:1</td>
      <td>NS:EDG</td>
    </tr>
    <tr>
      <th>2017-12-01 18:06:03.633</th>
      <td>2017-12-01 18:06:03.633</td>
      <td>Moody's Affirms Seven Classes of GSMS 2016-GS4</td>
      <td>urn:newsml:reuters.com:20171201:nMDY7wGNTP:1</td>
      <td>NS:RTRS</td>
    </tr>
    <tr>
      <th>2017-12-01 13:42:15.533</th>
      <td>2017-12-01 13:42:15.533</td>
      <td>UNICOM Global announces day one support for IB...</td>
      <td>urn:newsml:reuters.com:20171201:nNRA4znj7a:1</td>
      <td>NS:ENPNWS</td>
    </tr>
    <tr>
      <th>2017-12-01 05:49:15.000</th>
      <td>2017-12-01 05:49:20.672</td>
      <td>(EN) International Business Machines Corp Quar...</td>
      <td>urn:newsml:reuters.com:20171201:nGLF7z9y0r:2</td>
      <td>NS:GLFILE</td>
    </tr>
    <tr>
      <th>2017-12-01 00:15:03.143</th>
      <td>2017-12-01 00:15:03.143</td>
      <td>Renewal Of Ibm Websphere License</td>
      <td>urn:newsml:reuters.com:20171201:nNRA4zhy3w:1</td>
      <td>NS:ECLTND</td>
    </tr>
    <tr>
      <th>2017-11-30 23:12:26.054</th>
      <td>2017-11-30 23:12:26.054</td>
      <td>Mixed Contract For The Supply And Update Of Ve...</td>
      <td>urn:newsml:reuters.com:20171130:nNRA4zhhqh:1</td>
      <td>NS:ECLCTA</td>
    </tr>
    <tr>
      <th>2017-11-30 23:04:42.001</th>
      <td>2017-11-30 23:04:42.001</td>
      <td>Reuters Insider - Famed tech investor says Fac...</td>
      <td>urn:newsml:reuters.com:20171130:nRTV71mT4s:1</td>
      <td>NS:CNBC</td>
    </tr>
    <tr>
      <th>2017-11-30 22:39:51.254</th>
      <td>2017-11-30 22:39:51.254</td>
      <td>Reuters Insider - Three Dow stocks that could ...</td>
      <td>urn:newsml:reuters.com:20171130:nRTV4nyRRM:1</td>
      <td>NS:CNBC</td>
    </tr>
    <tr>
      <th>2017-11-30 20:45:10.059</th>
      <td>2017-11-30 20:45:10.059</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 447500.0 SHARES O...</td>
      <td>urn:newsml:reuters.com:20171130:nZHN0B668Y:1</td>
      <td>NS:RTRS</td>
    </tr>
    <tr>
      <th>2017-11-30 14:23:26.525</th>
      <td>2017-11-30 14:23:26.525</td>
      <td>IBM-Are mainframe customers confident their se...</td>
      <td>urn:newsml:reuters.com:20171130:nNRA4zcuml:1</td>
      <td>NS:ENPNWS</td>
    </tr>
    <tr>
      <th>2017-11-30 14:20:29.409</th>
      <td>2017-11-30 14:20:29.409</td>
      <td>Reuters Insider - Uptake CEO: Using predictive...</td>
      <td>urn:newsml:reuters.com:20171130:nRTV3k4vk3:1</td>
      <td>NS:CNBC</td>
    </tr>
    <tr>
      <th>2017-11-30 13:02:43.077</th>
      <td>2017-11-30 13:02:43.077</td>
      <td>BACK TO THE OFFICE Workplace IBM pioneered wor...</td>
      <td>urn:newsml:reuters.com:20171130:nNRA4zchgb:1</td>
      <td>NS:AUSFIN</td>
    </tr>
    <tr>
      <th>2017-11-30 00:10:18.117</th>
      <td>2017-11-30 00:10:18.117</td>
      <td>Robotics enables DBS to free up 25,000 man hou...</td>
      <td>urn:newsml:reuters.com:20171130:nNRA4z7j3u:1</td>
      <td>NS:BSNTMS</td>
    </tr>
    <tr>
      <th>2017-11-29 20:45:04.053</th>
      <td>2017-11-29 20:45:04.053</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 84000.0 SHARES ON...</td>
      <td>urn:newsml:reuters.com:20171129:nZHN0B6542:1</td>
      <td>NS:RTRS</td>
    </tr>
    <tr>
      <th>2017-11-29 18:20:32.000</th>
      <td>2017-11-29 18:28:44.000</td>
      <td>DJ 5 High-Yielding Stocks Ripe for the Picking...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW201902:2</td>
      <td>NS:DJN</td>
    </tr>
    <tr>
      <th>2017-11-29 17:42:32.000</th>
      <td>2017-11-29 17:42:32.000</td>
      <td>Update: IBM Server Responsible For Macy's Blac...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW2017E2:1</td>
      <td>NS:DJN</td>
    </tr>
    <tr>
      <th>2017-11-29 16:18:28.000</th>
      <td>2017-11-29 16:18:28.000</td>
      <td>Update: IBM Server Responsible For Macy's Blac...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW201537:1</td>
      <td>NS:DJN</td>
    </tr>
    <tr>
      <th>2017-11-29 16:09:37.000</th>
      <td>2017-11-29 16:09:56.000</td>
      <td>Update: IBM Server Responsible For Macy's Blac...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW2014ED:2</td>
      <td>NS:DJN</td>
    </tr>
    <tr>
      <th>2017-11-29 16:08:44.000</th>
      <td>2017-11-29 16:09:48.000</td>
      <td>DJ IBM Server Responsible For Macy's Black Fri...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW2014E8:2</td>
      <td>NS:DJN</td>
    </tr>
    <tr>
      <th>2017-11-29 16:03:31.497</th>
      <td>2017-11-29 16:03:31.497</td>
      <td>Conditions For AI Success: Discipline, Data, A...</td>
      <td>urn:newsml:reuters.com:20171129:nNRA4z4vjf:1</td>
      <td>NS:ABVLAW</td>
    </tr>
    <tr>
      <th>2017-11-29 13:52:12.000</th>
      <td>2017-11-29 13:52:12.000</td>
      <td>UPDATE 1-Munich Re's Ergo drops plan to sell r...</td>
      <td>urn:newsml:reuters.com:20171129:nL8N1NZ384:2</td>
      <td>NS:RTRS</td>
    </tr>
    <tr>
      <th>2017-11-29 11:36:34.130</th>
      <td>2017-11-29 11:36:34.130</td>
      <td>INTERVIEW: IBM advises SEE companies how to pr...</td>
      <td>urn:newsml:reuters.com:20171129:nSEEJkW0na:1</td>
      <td>NS:SEE</td>
    </tr>
    <tr>
      <th>2017-11-29 09:45:22.906</th>
      <td>2017-11-29 09:45:22.906</td>
      <td>Grammys 2018: Lady Gaga 'grateful' for two nom...</td>
      <td>urn:newsml:reuters.com:20171129:nNRA4z27k4:1</td>
      <td>NS:ASNEWS</td>
    </tr>
    <tr>
      <th>2017-11-29 00:45:51.955</th>
      <td>2017-11-29 00:45:51.955</td>
      <td>P/f S.s. Railing To Staircase Along With Balan...</td>
      <td>urn:newsml:reuters.com:20171129:nNRA4yy9zn:1</td>
      <td>NS:ECLTND</td>
    </tr>
    <tr>
      <th>2017-11-28 23:46:21.362</th>
      <td>2017-11-28 23:46:21.362</td>
      <td>Annual Repair &amp; Maintenance Operation To Non R...</td>
      <td>urn:newsml:reuters.com:20171128:nNRA4yy0l5:1</td>
      <td>NS:ECLTND</td>
    </tr>
    <tr>
      <th>2017-11-28 23:45:14.000</th>
      <td>2017-11-28 23:45:14.000</td>
      <td>DJ IBM's (IBM) Management Presents at 21st Ann...</td>
      <td>urn:newsml:reuters.com:20171128:nDJW1020F2:1</td>
      <td>NS:DJN</td>
    </tr>
    <tr>
      <th>2017-11-28 20:45:06.668</th>
      <td>2017-11-28 20:45:06.668</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 205700.0 SHARES O...</td>
      <td>urn:newsml:reuters.com:20171128:nZHN0B647K:1</td>
      <td>NS:RTRS</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-11-20 21:43:00.418</th>
      <td>2017-11-20 21:43:00.418</td>
      <td>Reuters Insider - Trading at Noon: Another chi...</td>
      <td>urn:newsml:reuters.com:20171120:nRTV1Tl9t2:1</td>
      <td>NS:RTRS</td>
    </tr>
    <tr>
      <th>2017-11-20 21:16:47.576</th>
      <td>2017-11-20 21:16:47.576</td>
      <td>Reuters Insider - Is the bottom in for IBM &amp; G...</td>
      <td>urn:newsml:reuters.com:20171120:nRTVbKpGhB:1</td>
      <td>NS:CNBC</td>
    </tr>
    <tr>
      <th>2017-11-20 19:50:58.000</th>
      <td>2017-11-20 19:51:58.000</td>
      <td>Update: Warren Buffett Buys Apple And Sells IB...</td>
      <td>urn:newsml:reuters.com:20171120:nDJWT016BF:2</td>
      <td>NS:DJN</td>
    </tr>
    <tr>
      <th>2017-11-20 15:15:25.490</th>
      <td>2017-11-20 15:15:25.490</td>
      <td>Reuters Insider - IBM leads Dow at the open</td>
      <td>urn:newsml:reuters.com:20171120:nRTV1SsXJy:1</td>
      <td>NS:CNBC</td>
    </tr>
    <tr>
      <th>2017-11-20 13:09:33.848</th>
      <td>2017-11-20 13:09:33.848</td>
      <td>Deutsche Bank partners with IBM for block-chai...</td>
      <td>urn:newsml:reuters.com:20171120:nNRA4x6kch:1</td>
      <td>NS:ENPNWS</td>
    </tr>
    <tr>
      <th>2017-11-20 11:43:00.045</th>
      <td>2017-11-20 11:43:00.045</td>
      <td>Reuters Insider - America has an infertility p...</td>
      <td>urn:newsml:reuters.com:20171120:nRTVc57Gfc:1</td>
      <td>NS:CNBC</td>
    </tr>
    <tr>
      <th>2017-11-20 08:45:13.875</th>
      <td>2017-11-20 08:45:13.875</td>
      <td>Soundbites: Dubai's Anita Williams back with n...</td>
      <td>urn:newsml:reuters.com:20171120:nNRA4x4jtl:1</td>
      <td>NS:GULNEW</td>
    </tr>
    <tr>
      <th>2017-11-19 21:22:30.000</th>
      <td>2017-11-19 21:22:30.000</td>
      <td>IBM could be set for gains after long slump -B...</td>
      <td>urn:newsml:reuters.com:20171119:nL1N1NP0L8:4</td>
      <td>NS:RTRS</td>
    </tr>
    <tr>
      <th>2017-11-18 23:52:55.324</th>
      <td>2017-11-18 23:52:55.324</td>
      <td>Comprehensive Amc Of Ibm Blade Center S Server</td>
      <td>urn:newsml:reuters.com:20171118:nNRA4wxol3:1</td>
      <td>NS:ECLTND</td>
    </tr>
    <tr>
      <th>2017-11-18 11:00:34.000</th>
      <td>2017-11-18 11:00:34.000</td>
      <td>DJ IBM: Blue Chip at a Bargain Price -- Barron's</td>
      <td>urn:newsml:reuters.com:20171118:nDJWR0010D:1</td>
      <td>NS:DJN</td>
    </tr>
    <tr>
      <th>2017-11-18 11:00:34.000</th>
      <td>2017-11-18 11:00:34.000</td>
      <td>DJ IBM: Blue Chip at a Bargain Price</td>
      <td>urn:newsml:reuters.com:20171118:nDJWR0010C:1</td>
      <td>NS:DJN</td>
    </tr>
    <tr>
      <th>2017-11-18 05:04:55.000</th>
      <td>2017-11-18 05:04:55.000</td>
      <td>DJ IBM: Bargain Blue Chip -- Barrons.com</td>
      <td>urn:newsml:reuters.com:20171118:nDJWR00003:1</td>
      <td>NS:DJN</td>
    </tr>
    <tr>
      <th>2017-11-18 04:50:21.836</th>
      <td>2017-11-18 04:50:21.836</td>
      <td>Supply And Delivery Of Ibm Spss Statistics Pre...</td>
      <td>urn:newsml:reuters.com:20171118:nNRA4wrzc7:1</td>
      <td>NS:ECLCTA</td>
    </tr>
    <tr>
      <th>2017-11-17 21:52:47.180</th>
      <td>2017-11-17 21:52:47.180</td>
      <td>Addressing a cybersecurity skills shortage wit...</td>
      <td>urn:newsml:reuters.com:20171117:nNRA4wljue:1</td>
      <td>NS:GLOBML</td>
    </tr>
    <tr>
      <th>2017-11-17 20:45:05.071</th>
      <td>2017-11-17 20:45:05.071</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 138500.0 SHARES O...</td>
      <td>urn:newsml:reuters.com:20171117:nZHN0B5T92:1</td>
      <td>NS:RTRS</td>
    </tr>
    <tr>
      <th>2017-11-17 18:28:17.260</th>
      <td>2017-11-17 18:28:17.260</td>
      <td>IBM ST: eye 136.2</td>
      <td>urn:newsml:reuters.com:20171117:nGUR5sxLFT:1</td>
      <td>NS:GURU</td>
    </tr>
    <tr>
      <th>2017-11-17 16:23:20.614</th>
      <td>2017-11-17 16:23:20.614</td>
      <td>Reuters Insider - Watch one man completely cra...</td>
      <td>urn:newsml:reuters.com:20171117:nRTV9j9KwX:1</td>
      <td>NS:CNBC</td>
    </tr>
    <tr>
      <th>2017-11-17 05:34:49.483</th>
      <td>2017-11-17 05:34:49.483</td>
      <td>Maintenance Service Of The Certified Datebase ...</td>
      <td>urn:newsml:reuters.com:20171117:nNRA4wib88:1</td>
      <td>NS:ECLCTA</td>
    </tr>
    <tr>
      <th>2017-11-17 00:48:31.862</th>
      <td>2017-11-17 00:48:31.862</td>
      <td>3 Factors that Defend IBM ETFs From Berkshire'...</td>
      <td>urn:newsml:reuters.com:20171117:nNRA4wgw0i:1</td>
      <td>NS:ZACKSC</td>
    </tr>
    <tr>
      <th>2017-11-17 00:31:27.680</th>
      <td>2017-11-17 00:31:27.680</td>
      <td>Reuters Insider - Cramer Remix: Warren Buffett...</td>
      <td>urn:newsml:reuters.com:20171117:nRTV1kKKFV:1</td>
      <td>NS:CNBC</td>
    </tr>
    <tr>
      <th>2017-11-17 00:09:29.826</th>
      <td>2017-11-17 00:09:29.826</td>
      <td>Reuters Insider - Cramer: Thank Wal-Mart and C...</td>
      <td>urn:newsml:reuters.com:20171117:nRTV4Ntxsn:1</td>
      <td>NS:CNBC</td>
    </tr>
    <tr>
      <th>2017-11-16 15:44:51.000</th>
      <td>2017-11-16 22:39:39.000</td>
      <td>UPDATE 2-IBM urged to avoid working on 'extrem...</td>
      <td>urn:newsml:reuters.com:20171116:nL1N1NM10X:2</td>
      <td>NS:RTRS</td>
    </tr>
    <tr>
      <th>2017-11-16 20:45:03.579</th>
      <td>2017-11-16 20:45:03.579</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 124000.0 SHARES O...</td>
      <td>urn:newsml:reuters.com:20171116:nZHN0B5S4D:1</td>
      <td>NS:RTRS</td>
    </tr>
    <tr>
      <th>2017-11-16 17:54:58.928</th>
      <td>2017-11-16 17:54:58.928</td>
      <td>INTERNATIONAL BUSINESS MACHINES CORP SEC Filin...</td>
      <td>urn:newsml:reuters.com:20171116:nEOL641hhv:1</td>
      <td>NS:EDG</td>
    </tr>
    <tr>
      <th>2017-11-16 17:54:28.684</th>
      <td>2017-11-16 17:54:28.684</td>
      <td>INTERNATIONAL BUSINESS MACHINES CORP SEC Filin...</td>
      <td>urn:newsml:reuters.com:20171116:nEOL1vVXRc:1</td>
      <td>NS:EDG</td>
    </tr>
    <tr>
      <th>2017-11-16 15:00:00.150</th>
      <td>2017-11-16 15:00:00.150</td>
      <td>Optoro Welcomes Jim Kelly as New EVP, Business...</td>
      <td>urn:newsml:reuters.com:20171116:nMKW198mFa:1</td>
      <td>NS:MKW</td>
    </tr>
    <tr>
      <th>2017-11-16 13:30:54.000</th>
      <td>2017-11-16 13:39:57.000</td>
      <td>Press Release: Fusion Genomics Turns to IBM Cl...</td>
      <td>urn:newsml:reuters.com:20171116:nDJWP00E68:2</td>
      <td>NS:DJN</td>
    </tr>
    <tr>
      <th>2017-11-16 13:30:53.372</th>
      <td>2017-11-16 13:30:53.372</td>
      <td>Fusion Genomics Turns to IBM Cloud to Help Sup...</td>
      <td>urn:newsml:reuters.com:20171116:nCNWh74gca:1</td>
      <td>NS:CNW</td>
    </tr>
    <tr>
      <th>2017-11-16 13:30:00.000</th>
      <td>2017-11-16 13:30:00.000</td>
      <td>Rights groups pressure IBM to renounce interes...</td>
      <td>urn:newsml:reuters.com:20171116:nL1N1NL22J:2</td>
      <td>NS:RTRS</td>
    </tr>
    <tr>
      <th>2017-11-16 13:00:53.294</th>
      <td>2017-11-16 13:00:53.294</td>
      <td>BMC Mainframe Solutions Accelerate Secure Digi...</td>
      <td>urn:newsml:reuters.com:20171116:nPnKbsbva:1</td>
      <td>NS:PRN</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>



I will just add 3 new columns which we will need to store some variables in later.


```python
df['Polarity'] = np.nan
df['Subjectivity'] = np.nan
df['Score'] = np.nan
```

So we have our frame with the most recent 100 news headline items. The headline is stored in the **text** column and the storyID which we will now use to pull down the actual articles themselves, is stored in the **storyID** column. 

We will now iterate through the headline dataframe and pull down the news articles using the second of our news API calls, get_news_story. We simply pass the **storyID** to this API call and we are returned a HTML representation of the article - which allows you to render them nicely etc - however for our purposes we want to strip the HTML tags etc out and just be left with the plain text - as we dont want to analyse HTML tags for sentiment. We will do this using the excellent **BeautifulSoup** package.

Once we have the text of these articles we can pass them to our sentiment engine which will give us a sentiment score for each article. So what is our sentiment engine? We will be using the simple **TextBlob** package to demo a rudimentary process to show you how things work. **TextBlob** is a higher level abstraction package that sits on top of **NLTK** (Natural Language Toolkit) which is a widely used package for this type of task. 

**NLTK** is quite a complex package which gives you a lot of control over the whole analytical process - but the cost of that is complexity and required knowledge of the steps invloved. **TextBlob** shields us from this complexity, but we should at some stage understand what is going on under the hood. Thankfully there is plenty of information to guide us in this. We will be implementing the default **PatternAnalyzer** which is based on the popular **Pattern** library though there is also a **NaiveBayesAnalyzer** which is a **NLTK** classifier based on a movie review corpus. 

All of this can be achieved in just a few lines of code. This is quite a dense codeblock - so I have commented the key steps.  


```python
for idx, storyId in enumerate(df['storyId'].values):  #for each row in our df dataframe
    newsText = ek.get_news_story(storyId) #get the news story
    if newsText:
        soup = BeautifulSoup(newsText,"lxml") #create a BeautifulSoup object from our HTML news article
        sentA = TextBlob(soup.get_text()) #pass the text only article to TextBlob to anaylse
        df['Polarity'].iloc[idx] = sentA.sentiment.polarity #write sentiment polarity back to df
        df['Subjectivity'].iloc[idx] = sentA.sentiment.subjectivity #write sentiment subjectivity score back to df
        if sentA.sentiment.polarity >= 0.05: # attribute bucket to sentiment polartiy
            score = 'positive'
        elif  -.05 < sentA.sentiment.polarity < 0.05:
            score = 'neutral'
        else:
            score = 'negative'
        df['Score'].iloc[idx] = score #write score back to df
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>versionCreated</th>
      <th>text</th>
      <th>storyId</th>
      <th>sourceCode</th>
      <th>Polarity</th>
      <th>Subjectivity</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-12-01 23:11:47.374</th>
      <td>2017-12-01 23:11:47.374</td>
      <td>Reuters Insider - FM Final Trade: HAL, TWTR &amp; ...</td>
      <td>urn:newsml:reuters.com:20171201:nRTV8KBb1N:1</td>
      <td>NS:CNBC</td>
      <td>0.066667</td>
      <td>0.566667</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-12-01 19:19:20.279</th>
      <td>2017-12-01 19:19:20.279</td>
      <td>IBM ST: the upside prevails as long as 150.2 i...</td>
      <td>urn:newsml:reuters.com:20171201:nGUR2R6xQ:1</td>
      <td>NS:GURU</td>
      <td>0.055260</td>
      <td>0.320844</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-12-01 18:12:41.143</th>
      <td>2017-12-01 18:12:41.143</td>
      <td>INTERNATIONAL BUSINESS MACHINES CORP SEC Filin...</td>
      <td>urn:newsml:reuters.com:20171201:nEOL6JJfRT:1</td>
      <td>NS:EDG</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-12-01 18:12:41.019</th>
      <td>2017-12-01 18:12:41.019</td>
      <td>INTERNATIONAL BUSINESS MACHINES CORP SEC Filin...</td>
      <td>urn:newsml:reuters.com:20171201:nEOL3YHcVY:1</td>
      <td>NS:EDG</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-12-01 18:06:03.633</th>
      <td>2017-12-01 18:06:03.633</td>
      <td>Moody's Affirms Seven Classes of GSMS 2016-GS4</td>
      <td>urn:newsml:reuters.com:20171201:nMDY7wGNTP:1</td>
      <td>NS:RTRS</td>
      <td>0.175000</td>
      <td>0.325000</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-12-01 13:42:15.533</th>
      <td>2017-12-01 13:42:15.533</td>
      <td>UNICOM Global announces day one support for IB...</td>
      <td>urn:newsml:reuters.com:20171201:nNRA4znj7a:1</td>
      <td>NS:ENPNWS</td>
      <td>0.111547</td>
      <td>0.246926</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-12-01 05:49:15.000</th>
      <td>2017-12-01 05:49:20.672</td>
      <td>(EN) International Business Machines Corp Quar...</td>
      <td>urn:newsml:reuters.com:20171201:nGLF7z9y0r:2</td>
      <td>NS:GLFILE</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-12-01 00:15:03.143</th>
      <td>2017-12-01 00:15:03.143</td>
      <td>Renewal Of Ibm Websphere License</td>
      <td>urn:newsml:reuters.com:20171201:nNRA4zhy3w:1</td>
      <td>NS:ECLTND</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-30 23:12:26.054</th>
      <td>2017-11-30 23:12:26.054</td>
      <td>Mixed Contract For The Supply And Update Of Ve...</td>
      <td>urn:newsml:reuters.com:20171130:nNRA4zhhqh:1</td>
      <td>NS:ECLCTA</td>
      <td>0.050000</td>
      <td>0.500000</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-30 23:04:42.001</th>
      <td>2017-11-30 23:04:42.001</td>
      <td>Reuters Insider - Famed tech investor says Fac...</td>
      <td>urn:newsml:reuters.com:20171130:nRTV71mT4s:1</td>
      <td>NS:CNBC</td>
      <td>0.500000</td>
      <td>0.200000</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-30 22:39:51.254</th>
      <td>2017-11-30 22:39:51.254</td>
      <td>Reuters Insider - Three Dow stocks that could ...</td>
      <td>urn:newsml:reuters.com:20171130:nRTV4nyRRM:1</td>
      <td>NS:CNBC</td>
      <td>0.187500</td>
      <td>0.425000</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-30 20:45:10.059</th>
      <td>2017-11-30 20:45:10.059</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 447500.0 SHARES O...</td>
      <td>urn:newsml:reuters.com:20171130:nZHN0B668Y:1</td>
      <td>NS:RTRS</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-30 14:23:26.525</th>
      <td>2017-11-30 14:23:26.525</td>
      <td>IBM-Are mainframe customers confident their se...</td>
      <td>urn:newsml:reuters.com:20171130:nNRA4zcuml:1</td>
      <td>NS:ENPNWS</td>
      <td>0.152041</td>
      <td>0.410031</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-30 14:20:29.409</th>
      <td>2017-11-30 14:20:29.409</td>
      <td>Reuters Insider - Uptake CEO: Using predictive...</td>
      <td>urn:newsml:reuters.com:20171130:nRTV3k4vk3:1</td>
      <td>NS:CNBC</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-30 13:02:43.077</th>
      <td>2017-11-30 13:02:43.077</td>
      <td>BACK TO THE OFFICE Workplace IBM pioneered wor...</td>
      <td>urn:newsml:reuters.com:20171130:nNRA4zchgb:1</td>
      <td>NS:AUSFIN</td>
      <td>0.060436</td>
      <td>0.398085</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-30 00:10:18.117</th>
      <td>2017-11-30 00:10:18.117</td>
      <td>Robotics enables DBS to free up 25,000 man hou...</td>
      <td>urn:newsml:reuters.com:20171130:nNRA4z7j3u:1</td>
      <td>NS:BSNTMS</td>
      <td>0.008420</td>
      <td>0.420476</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-29 20:45:04.053</th>
      <td>2017-11-29 20:45:04.053</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 84000.0 SHARES ON...</td>
      <td>urn:newsml:reuters.com:20171129:nZHN0B6542:1</td>
      <td>NS:RTRS</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-29 18:20:32.000</th>
      <td>2017-11-29 18:28:44.000</td>
      <td>DJ 5 High-Yielding Stocks Ripe for the Picking...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW201902:2</td>
      <td>NS:DJN</td>
      <td>0.058534</td>
      <td>0.385188</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-29 17:42:32.000</th>
      <td>2017-11-29 17:42:32.000</td>
      <td>Update: IBM Server Responsible For Macy's Blac...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW2017E2:1</td>
      <td>NS:DJN</td>
      <td>0.108194</td>
      <td>0.466389</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-29 16:18:28.000</th>
      <td>2017-11-29 16:18:28.000</td>
      <td>Update: IBM Server Responsible For Macy's Blac...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW201537:1</td>
      <td>NS:DJN</td>
      <td>0.057253</td>
      <td>0.440432</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-29 16:09:37.000</th>
      <td>2017-11-29 16:09:56.000</td>
      <td>Update: IBM Server Responsible For Macy's Blac...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW2014ED:2</td>
      <td>NS:DJN</td>
      <td>0.057253</td>
      <td>0.440432</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-29 16:08:44.000</th>
      <td>2017-11-29 16:09:48.000</td>
      <td>DJ IBM Server Responsible For Macy's Black Fri...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW2014E8:2</td>
      <td>NS:DJN</td>
      <td>0.057253</td>
      <td>0.440432</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-29 16:03:31.497</th>
      <td>2017-11-29 16:03:31.497</td>
      <td>Conditions For AI Success: Discipline, Data, A...</td>
      <td>urn:newsml:reuters.com:20171129:nNRA4z4vjf:1</td>
      <td>NS:ABVLAW</td>
      <td>0.089723</td>
      <td>0.406275</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-29 13:52:12.000</th>
      <td>2017-11-29 13:52:12.000</td>
      <td>UPDATE 1-Munich Re's Ergo drops plan to sell r...</td>
      <td>urn:newsml:reuters.com:20171129:nL8N1NZ384:2</td>
      <td>NS:RTRS</td>
      <td>0.051498</td>
      <td>0.315673</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-29 11:36:34.130</th>
      <td>2017-11-29 11:36:34.130</td>
      <td>INTERVIEW: IBM advises SEE companies how to pr...</td>
      <td>urn:newsml:reuters.com:20171129:nSEEJkW0na:1</td>
      <td>NS:SEE</td>
      <td>0.134661</td>
      <td>0.383414</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-29 09:45:22.906</th>
      <td>2017-11-29 09:45:22.906</td>
      <td>Grammys 2018: Lady Gaga 'grateful' for two nom...</td>
      <td>urn:newsml:reuters.com:20171129:nNRA4z27k4:1</td>
      <td>NS:ASNEWS</td>
      <td>0.040990</td>
      <td>0.510119</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-29 00:45:51.955</th>
      <td>2017-11-29 00:45:51.955</td>
      <td>P/f S.s. Railing To Staircase Along With Balan...</td>
      <td>urn:newsml:reuters.com:20171129:nNRA4yy9zn:1</td>
      <td>NS:ECLTND</td>
      <td>0.000000</td>
      <td>0.158333</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-28 23:46:21.362</th>
      <td>2017-11-28 23:46:21.362</td>
      <td>Annual Repair &amp; Maintenance Operation To Non R...</td>
      <td>urn:newsml:reuters.com:20171128:nNRA4yy0l5:1</td>
      <td>NS:ECLTND</td>
      <td>0.000000</td>
      <td>0.188889</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-28 23:45:14.000</th>
      <td>2017-11-28 23:45:14.000</td>
      <td>DJ IBM's (IBM) Management Presents at 21st Ann...</td>
      <td>urn:newsml:reuters.com:20171128:nDJW1020F2:1</td>
      <td>NS:DJN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-28 20:45:06.668</th>
      <td>2017-11-28 20:45:06.668</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 205700.0 SHARES O...</td>
      <td>urn:newsml:reuters.com:20171128:nZHN0B647K:1</td>
      <td>NS:RTRS</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-11-20 21:43:00.418</th>
      <td>2017-11-20 21:43:00.418</td>
      <td>Reuters Insider - Trading at Noon: Another chi...</td>
      <td>urn:newsml:reuters.com:20171120:nRTV1Tl9t2:1</td>
      <td>NS:RTRS</td>
      <td>0.125000</td>
      <td>0.562500</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-20 21:16:47.576</th>
      <td>2017-11-20 21:16:47.576</td>
      <td>Reuters Insider - Is the bottom in for IBM &amp; G...</td>
      <td>urn:newsml:reuters.com:20171120:nRTVbKpGhB:1</td>
      <td>NS:CNBC</td>
      <td>-0.191667</td>
      <td>0.533333</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2017-11-20 19:50:58.000</th>
      <td>2017-11-20 19:51:58.000</td>
      <td>Update: Warren Buffett Buys Apple And Sells IB...</td>
      <td>urn:newsml:reuters.com:20171120:nDJWT016BF:2</td>
      <td>NS:DJN</td>
      <td>0.172727</td>
      <td>0.571591</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-20 15:15:25.490</th>
      <td>2017-11-20 15:15:25.490</td>
      <td>Reuters Insider - IBM leads Dow at the open</td>
      <td>urn:newsml:reuters.com:20171120:nRTV1SsXJy:1</td>
      <td>NS:CNBC</td>
      <td>0.050000</td>
      <td>0.200000</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-20 13:09:33.848</th>
      <td>2017-11-20 13:09:33.848</td>
      <td>Deutsche Bank partners with IBM for block-chai...</td>
      <td>urn:newsml:reuters.com:20171120:nNRA4x6kch:1</td>
      <td>NS:ENPNWS</td>
      <td>0.110985</td>
      <td>0.424116</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-20 11:43:00.045</th>
      <td>2017-11-20 11:43:00.045</td>
      <td>Reuters Insider - America has an infertility p...</td>
      <td>urn:newsml:reuters.com:20171120:nRTVc57Gfc:1</td>
      <td>NS:CNBC</td>
      <td>-0.077778</td>
      <td>0.194444</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2017-11-20 08:45:13.875</th>
      <td>2017-11-20 08:45:13.875</td>
      <td>Soundbites: Dubai's Anita Williams back with n...</td>
      <td>urn:newsml:reuters.com:20171120:nNRA4x4jtl:1</td>
      <td>NS:GULNEW</td>
      <td>0.196711</td>
      <td>0.454477</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-19 21:22:30.000</th>
      <td>2017-11-19 21:22:30.000</td>
      <td>IBM could be set for gains after long slump -B...</td>
      <td>urn:newsml:reuters.com:20171119:nL1N1NP0L8:4</td>
      <td>NS:RTRS</td>
      <td>0.082986</td>
      <td>0.196181</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-18 23:52:55.324</th>
      <td>2017-11-18 23:52:55.324</td>
      <td>Comprehensive Amc Of Ibm Blade Center S Server</td>
      <td>urn:newsml:reuters.com:20171118:nNRA4wxol3:1</td>
      <td>NS:ECLTND</td>
      <td>-0.100000</td>
      <td>0.100000</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2017-11-18 11:00:34.000</th>
      <td>2017-11-18 11:00:34.000</td>
      <td>DJ IBM: Blue Chip at a Bargain Price -- Barron's</td>
      <td>urn:newsml:reuters.com:20171118:nDJWR0010D:1</td>
      <td>NS:DJN</td>
      <td>0.111388</td>
      <td>0.402399</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-18 11:00:34.000</th>
      <td>2017-11-18 11:00:34.000</td>
      <td>DJ IBM: Blue Chip at a Bargain Price</td>
      <td>urn:newsml:reuters.com:20171118:nDJWR0010C:1</td>
      <td>NS:DJN</td>
      <td>0.111388</td>
      <td>0.402399</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-18 05:04:55.000</th>
      <td>2017-11-18 05:04:55.000</td>
      <td>DJ IBM: Bargain Blue Chip -- Barrons.com</td>
      <td>urn:newsml:reuters.com:20171118:nDJWR00003:1</td>
      <td>NS:DJN</td>
      <td>0.111388</td>
      <td>0.402399</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-18 04:50:21.836</th>
      <td>2017-11-18 04:50:21.836</td>
      <td>Supply And Delivery Of Ibm Spss Statistics Pre...</td>
      <td>urn:newsml:reuters.com:20171118:nNRA4wrzc7:1</td>
      <td>NS:ECLCTA</td>
      <td>-0.071429</td>
      <td>0.214286</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2017-11-17 21:52:47.180</th>
      <td>2017-11-17 21:52:47.180</td>
      <td>Addressing a cybersecurity skills shortage wit...</td>
      <td>urn:newsml:reuters.com:20171117:nNRA4wljue:1</td>
      <td>NS:GLOBML</td>
      <td>0.160388</td>
      <td>0.463328</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-17 20:45:05.071</th>
      <td>2017-11-17 20:45:05.071</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 138500.0 SHARES O...</td>
      <td>urn:newsml:reuters.com:20171117:nZHN0B5T92:1</td>
      <td>NS:RTRS</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-17 18:28:17.260</th>
      <td>2017-11-17 18:28:17.260</td>
      <td>IBM ST: eye 136.2</td>
      <td>urn:newsml:reuters.com:20171117:nGUR5sxLFT:1</td>
      <td>NS:GURU</td>
      <td>0.005921</td>
      <td>0.325057</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-17 16:23:20.614</th>
      <td>2017-11-17 16:23:20.614</td>
      <td>Reuters Insider - Watch one man completely cra...</td>
      <td>urn:newsml:reuters.com:20171117:nRTV9j9KwX:1</td>
      <td>NS:CNBC</td>
      <td>0.066667</td>
      <td>0.366667</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-17 05:34:49.483</th>
      <td>2017-11-17 05:34:49.483</td>
      <td>Maintenance Service Of The Certified Datebase ...</td>
      <td>urn:newsml:reuters.com:20171117:nNRA4wib88:1</td>
      <td>NS:ECLCTA</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-17 00:48:31.862</th>
      <td>2017-11-17 00:48:31.862</td>
      <td>3 Factors that Defend IBM ETFs From Berkshire'...</td>
      <td>urn:newsml:reuters.com:20171117:nNRA4wgw0i:1</td>
      <td>NS:ZACKSC</td>
      <td>0.135930</td>
      <td>0.462621</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-17 00:31:27.680</th>
      <td>2017-11-17 00:31:27.680</td>
      <td>Reuters Insider - Cramer Remix: Warren Buffett...</td>
      <td>urn:newsml:reuters.com:20171117:nRTV1kKKFV:1</td>
      <td>NS:CNBC</td>
      <td>0.100000</td>
      <td>0.300000</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-17 00:09:29.826</th>
      <td>2017-11-17 00:09:29.826</td>
      <td>Reuters Insider - Cramer: Thank Wal-Mart and C...</td>
      <td>urn:newsml:reuters.com:20171117:nRTV4Ntxsn:1</td>
      <td>NS:CNBC</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-16 15:44:51.000</th>
      <td>2017-11-16 22:39:39.000</td>
      <td>UPDATE 2-IBM urged to avoid working on 'extrem...</td>
      <td>urn:newsml:reuters.com:20171116:nL1N1NM10X:2</td>
      <td>NS:RTRS</td>
      <td>-0.014847</td>
      <td>0.479955</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-16 20:45:03.579</th>
      <td>2017-11-16 20:45:03.579</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 124000.0 SHARES O...</td>
      <td>urn:newsml:reuters.com:20171116:nZHN0B5S4D:1</td>
      <td>NS:RTRS</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-16 17:54:58.928</th>
      <td>2017-11-16 17:54:58.928</td>
      <td>INTERNATIONAL BUSINESS MACHINES CORP SEC Filin...</td>
      <td>urn:newsml:reuters.com:20171116:nEOL641hhv:1</td>
      <td>NS:EDG</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-16 17:54:28.684</th>
      <td>2017-11-16 17:54:28.684</td>
      <td>INTERNATIONAL BUSINESS MACHINES CORP SEC Filin...</td>
      <td>urn:newsml:reuters.com:20171116:nEOL1vVXRc:1</td>
      <td>NS:EDG</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-16 15:00:00.150</th>
      <td>2017-11-16 15:00:00.150</td>
      <td>Optoro Welcomes Jim Kelly as New EVP, Business...</td>
      <td>urn:newsml:reuters.com:20171116:nMKW198mFa:1</td>
      <td>NS:MKW</td>
      <td>0.306273</td>
      <td>0.415809</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-16 13:30:54.000</th>
      <td>2017-11-16 13:39:57.000</td>
      <td>Press Release: Fusion Genomics Turns to IBM Cl...</td>
      <td>urn:newsml:reuters.com:20171116:nDJWP00E68:2</td>
      <td>NS:DJN</td>
      <td>0.141170</td>
      <td>0.469894</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-16 13:30:53.372</th>
      <td>2017-11-16 13:30:53.372</td>
      <td>Fusion Genomics Turns to IBM Cloud to Help Sup...</td>
      <td>urn:newsml:reuters.com:20171116:nCNWh74gca:1</td>
      <td>NS:CNW</td>
      <td>0.187248</td>
      <td>0.475294</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2017-11-16 13:30:00.000</th>
      <td>2017-11-16 13:30:00.000</td>
      <td>Rights groups pressure IBM to renounce interes...</td>
      <td>urn:newsml:reuters.com:20171116:nL1N1NL22J:2</td>
      <td>NS:RTRS</td>
      <td>0.001837</td>
      <td>0.498600</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>2017-11-16 13:00:53.294</th>
      <td>2017-11-16 13:00:53.294</td>
      <td>BMC Mainframe Solutions Accelerate Secure Digi...</td>
      <td>urn:newsml:reuters.com:20171116:nPnKbsbva:1</td>
      <td>NS:PRN</td>
      <td>0.180128</td>
      <td>0.486111</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 7 columns</p>
</div>



Looking at our dataframe we can now see 3 new columns on the right, *Polarity*, *Subjectivity* and *Score*. As we have seen *Polarity* is the actual sentiment polarity returned from **TextBlob** (ranging from -1(negative) to +1(positive), *Subjectivity* is a measure (ranging from 0 to 1) where 0 is very objective and 1 is very subjective, and *Score* is simply a Positive, Negative or Neutral rating based on the strength of the polarities. 

We would now like to see what, if any, impact this news has had on the shareprice of **IBM**. There are many ways of doing this - but to make things simple, I would like to see what the average return is at various points in time **AFTER** the news has broken. I want to check if there are *aggregate differences* in the *average returns* from the Positive, Neutral and Negative buckets we created earlier.


```python
start = df['versionCreated'].min().replace(hour=0,minute=0,second=0,microsecond=0).strftime('%Y/%m/%d')
end = df['versionCreated'].max().replace(hour=0,minute=0,second=0,microsecond=0).strftime('%Y/%m/%d')
Minute = ek.get_timeseries(["IBM.N"], start_date=start, interval="minute")
Minute.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>IBM.N</th>
      <th>HIGH</th>
      <th>LOW</th>
      <th>OPEN</th>
      <th>CLOSE</th>
      <th>COUNT</th>
      <th>VOLUME</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-04 20:57:00</th>
      <td>161.73</td>
      <td>161.70</td>
      <td>161.70</td>
      <td>161.73</td>
      <td>28.0</td>
      <td>7529.0</td>
    </tr>
    <tr>
      <th>2018-01-04 20:58:00</th>
      <td>161.73</td>
      <td>161.65</td>
      <td>161.73</td>
      <td>161.72</td>
      <td>82.0</td>
      <td>14673.0</td>
    </tr>
    <tr>
      <th>2018-01-04 20:59:00</th>
      <td>161.78</td>
      <td>161.71</td>
      <td>161.73</td>
      <td>161.77</td>
      <td>162.0</td>
      <td>19367.0</td>
    </tr>
    <tr>
      <th>2018-01-04 21:00:00</th>
      <td>161.81</td>
      <td>161.63</td>
      <td>161.76</td>
      <td>161.67</td>
      <td>259.0</td>
      <td>50036.0</td>
    </tr>
    <tr>
      <th>2018-01-04 21:02:00</th>
      <td>161.70</td>
      <td>161.70</td>
      <td>161.70</td>
      <td>161.70</td>
      <td>1.0</td>
      <td>473227.0</td>
    </tr>
  </tbody>
</table>
</div>



We will need to create some new columns for the next part of this analysis.


```python
df['twoM'] = np.nan
df['fiveM'] = np.nan
df['tenM'] = np.nan
df['thirtyM'] = np.nan
df.head(2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>versionCreated</th>
      <th>text</th>
      <th>storyId</th>
      <th>sourceCode</th>
      <th>Polarity</th>
      <th>Subjectivity</th>
      <th>Score</th>
      <th>twoM</th>
      <th>fiveM</th>
      <th>tenM</th>
      <th>thirtyM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-12-01 23:11:47.374</th>
      <td>2017-12-01 23:11:47.374</td>
      <td>Reuters Insider - FM Final Trade: HAL, TWTR &amp; ...</td>
      <td>urn:newsml:reuters.com:20171201:nRTV8KBb1N:1</td>
      <td>NS:CNBC</td>
      <td>0.066667</td>
      <td>0.566667</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-12-01 19:19:20.279</th>
      <td>2017-12-01 19:19:20.279</td>
      <td>IBM ST: the upside prevails as long as 150.2 i...</td>
      <td>urn:newsml:reuters.com:20171201:nGUR2R6xQ:1</td>
      <td>NS:GURU</td>
      <td>0.055260</td>
      <td>0.320844</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



OK so I now just need to get the timestamp of each news item, truncate it to minute data (ie remove second and microsecond components) and get the base shareprice of **IBM** at that time, and at several itervals after that time, in our case *t+2 mins,t+5 mins, t+10 mins, t+30 mins*, calculating the % change for each interval. 

An important point to bear in mind here is that news can be generated at anytime - 24 hours a day - outside of normal market hours. So for news generated outside normal market hours for **IBM** in our case, we would have to wait until the next market opening to conduct our calculations. Of course there are a number of issues here concerning our ability to attribute price movement to our news item in isolation (basically we cannot). That said, there might be other ways of doing this - for example looking at **GDRs/ADRs** or surrogates etc - these are beyond the scope of this introductory article. In our example, these news items are simply discarded. 

We will now loop through each news item in the dataframe, calculate (where possible) and store the derived performance numbers in the columns we created earlier: twoM...thirtyM.


```python
for idx, newsDate in enumerate(df['versionCreated'].values):
    sTime = df['versionCreated'][idx]
    sTime = sTime.replace(second=0,microsecond=0)
    try:
        t0 = Minute.iloc[Minute.index.get_loc(sTime),2]
        df['twoM'][idx] = ((Minute.iloc[Minute.index.get_loc((sTime + datetime.timedelta(minutes=2))),3]/(t0)-1)*100)
        df['fiveM'][idx] = ((Minute.iloc[Minute.index.get_loc((sTime + datetime.timedelta(minutes=5))),3]/(t0)-1)*100)
        df['tenM'][idx] = ((Minute.iloc[Minute.index.get_loc((sTime + datetime.timedelta(minutes=10))),3]/(t0)-1)*100) 
        df['thirtyM'][idx] = ((Minute.iloc[Minute.index.get_loc((sTime + datetime.timedelta(minutes=30))),3]/(t0)-1)*100)
    except:
        pass
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>versionCreated</th>
      <th>text</th>
      <th>storyId</th>
      <th>sourceCode</th>
      <th>Polarity</th>
      <th>Subjectivity</th>
      <th>Score</th>
      <th>twoM</th>
      <th>fiveM</th>
      <th>tenM</th>
      <th>thirtyM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-12-01 23:11:47.374</th>
      <td>2017-12-01 23:11:47.374</td>
      <td>Reuters Insider - FM Final Trade: HAL, TWTR &amp; ...</td>
      <td>urn:newsml:reuters.com:20171201:nRTV8KBb1N:1</td>
      <td>NS:CNBC</td>
      <td>0.066667</td>
      <td>0.566667</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-12-01 19:19:20.279</th>
      <td>2017-12-01 19:19:20.279</td>
      <td>IBM ST: the upside prevails as long as 150.2 i...</td>
      <td>urn:newsml:reuters.com:20171201:nGUR2R6xQ:1</td>
      <td>NS:GURU</td>
      <td>0.055260</td>
      <td>0.320844</td>
      <td>positive</td>
      <td>0.071119</td>
      <td>0.084050</td>
      <td>0.000000</td>
      <td>-0.109911</td>
    </tr>
    <tr>
      <th>2017-12-01 18:12:41.143</th>
      <td>2017-12-01 18:12:41.143</td>
      <td>INTERNATIONAL BUSINESS MACHINES CORP SEC Filin...</td>
      <td>urn:newsml:reuters.com:20171201:nEOL6JJfRT:1</td>
      <td>NS:EDG</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>0.012944</td>
      <td>-0.090609</td>
      <td>-0.032360</td>
      <td>0.148858</td>
    </tr>
    <tr>
      <th>2017-12-01 18:12:41.019</th>
      <td>2017-12-01 18:12:41.019</td>
      <td>INTERNATIONAL BUSINESS MACHINES CORP SEC Filin...</td>
      <td>urn:newsml:reuters.com:20171201:nEOL3YHcVY:1</td>
      <td>NS:EDG</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>0.012944</td>
      <td>-0.090609</td>
      <td>-0.032360</td>
      <td>0.148858</td>
    </tr>
    <tr>
      <th>2017-12-01 18:06:03.633</th>
      <td>2017-12-01 18:06:03.633</td>
      <td>Moody's Affirms Seven Classes of GSMS 2016-GS4</td>
      <td>urn:newsml:reuters.com:20171201:nMDY7wGNTP:1</td>
      <td>NS:RTRS</td>
      <td>0.175000</td>
      <td>0.325000</td>
      <td>positive</td>
      <td>0.097238</td>
      <td>0.155581</td>
      <td>0.097238</td>
      <td>0.246337</td>
    </tr>
    <tr>
      <th>2017-12-01 13:42:15.533</th>
      <td>2017-12-01 13:42:15.533</td>
      <td>UNICOM Global announces day one support for IB...</td>
      <td>urn:newsml:reuters.com:20171201:nNRA4znj7a:1</td>
      <td>NS:ENPNWS</td>
      <td>0.111547</td>
      <td>0.246926</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-12-01 05:49:15.000</th>
      <td>2017-12-01 05:49:20.672</td>
      <td>(EN) International Business Machines Corp Quar...</td>
      <td>urn:newsml:reuters.com:20171201:nGLF7z9y0r:2</td>
      <td>NS:GLFILE</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-12-01 00:15:03.143</th>
      <td>2017-12-01 00:15:03.143</td>
      <td>Renewal Of Ibm Websphere License</td>
      <td>urn:newsml:reuters.com:20171201:nNRA4zhy3w:1</td>
      <td>NS:ECLTND</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-30 23:12:26.054</th>
      <td>2017-11-30 23:12:26.054</td>
      <td>Mixed Contract For The Supply And Update Of Ve...</td>
      <td>urn:newsml:reuters.com:20171130:nNRA4zhhqh:1</td>
      <td>NS:ECLCTA</td>
      <td>0.050000</td>
      <td>0.500000</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-30 23:04:42.001</th>
      <td>2017-11-30 23:04:42.001</td>
      <td>Reuters Insider - Famed tech investor says Fac...</td>
      <td>urn:newsml:reuters.com:20171130:nRTV71mT4s:1</td>
      <td>NS:CNBC</td>
      <td>0.500000</td>
      <td>0.200000</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-30 22:39:51.254</th>
      <td>2017-11-30 22:39:51.254</td>
      <td>Reuters Insider - Three Dow stocks that could ...</td>
      <td>urn:newsml:reuters.com:20171130:nRTV4nyRRM:1</td>
      <td>NS:CNBC</td>
      <td>0.187500</td>
      <td>0.425000</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-30 20:45:10.059</th>
      <td>2017-11-30 20:45:10.059</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 447500.0 SHARES O...</td>
      <td>urn:newsml:reuters.com:20171130:nZHN0B668Y:1</td>
      <td>NS:RTRS</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>0.052033</td>
      <td>0.156098</td>
      <td>0.266667</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-30 14:23:26.525</th>
      <td>2017-11-30 14:23:26.525</td>
      <td>IBM-Are mainframe customers confident their se...</td>
      <td>urn:newsml:reuters.com:20171130:nNRA4zcuml:1</td>
      <td>NS:ENPNWS</td>
      <td>0.152041</td>
      <td>0.410031</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-30 14:20:29.409</th>
      <td>2017-11-30 14:20:29.409</td>
      <td>Reuters Insider - Uptake CEO: Using predictive...</td>
      <td>urn:newsml:reuters.com:20171130:nRTV3k4vk3:1</td>
      <td>NS:CNBC</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>neutral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-30 13:02:43.077</th>
      <td>2017-11-30 13:02:43.077</td>
      <td>BACK TO THE OFFICE Workplace IBM pioneered wor...</td>
      <td>urn:newsml:reuters.com:20171130:nNRA4zchgb:1</td>
      <td>NS:AUSFIN</td>
      <td>0.060436</td>
      <td>0.398085</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-30 00:10:18.117</th>
      <td>2017-11-30 00:10:18.117</td>
      <td>Robotics enables DBS to free up 25,000 man hou...</td>
      <td>urn:newsml:reuters.com:20171130:nNRA4z7j3u:1</td>
      <td>NS:BSNTMS</td>
      <td>0.008420</td>
      <td>0.420476</td>
      <td>neutral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-29 20:45:04.053</th>
      <td>2017-11-29 20:45:04.053</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 84000.0 SHARES ON...</td>
      <td>urn:newsml:reuters.com:20171129:nZHN0B6542:1</td>
      <td>NS:RTRS</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>-0.019554</td>
      <td>-0.006518</td>
      <td>0.032590</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-29 18:20:32.000</th>
      <td>2017-11-29 18:28:44.000</td>
      <td>DJ 5 High-Yielding Stocks Ripe for the Picking...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW201902:2</td>
      <td>NS:DJN</td>
      <td>0.058534</td>
      <td>0.385188</td>
      <td>positive</td>
      <td>0.026144</td>
      <td>0.032680</td>
      <td>0.039216</td>
      <td>0.065359</td>
    </tr>
    <tr>
      <th>2017-11-29 17:42:32.000</th>
      <td>2017-11-29 17:42:32.000</td>
      <td>Update: IBM Server Responsible For Macy's Blac...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW2017E2:1</td>
      <td>NS:DJN</td>
      <td>0.108194</td>
      <td>0.466389</td>
      <td>positive</td>
      <td>0.039221</td>
      <td>0.039221</td>
      <td>0.039221</td>
      <td>0.013074</td>
    </tr>
    <tr>
      <th>2017-11-29 16:18:28.000</th>
      <td>2017-11-29 16:18:28.000</td>
      <td>Update: IBM Server Responsible For Macy's Blac...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW201537:1</td>
      <td>NS:DJN</td>
      <td>0.057253</td>
      <td>0.440432</td>
      <td>positive</td>
      <td>0.085201</td>
      <td>0.052432</td>
      <td>0.183510</td>
      <td>0.137633</td>
    </tr>
    <tr>
      <th>2017-11-29 16:09:37.000</th>
      <td>2017-11-29 16:09:56.000</td>
      <td>Update: IBM Server Responsible For Macy's Blac...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW2014ED:2</td>
      <td>NS:DJN</td>
      <td>0.057253</td>
      <td>0.440432</td>
      <td>positive</td>
      <td>0.006556</td>
      <td>0.032780</td>
      <td>0.085229</td>
      <td>0.190127</td>
    </tr>
    <tr>
      <th>2017-11-29 16:08:44.000</th>
      <td>2017-11-29 16:09:48.000</td>
      <td>DJ IBM Server Responsible For Macy's Black Fri...</td>
      <td>urn:newsml:reuters.com:20171129:nDJW2014E8:2</td>
      <td>NS:DJN</td>
      <td>0.057253</td>
      <td>0.440432</td>
      <td>positive</td>
      <td>0.006556</td>
      <td>0.032780</td>
      <td>0.085229</td>
      <td>0.190127</td>
    </tr>
    <tr>
      <th>2017-11-29 16:03:31.497</th>
      <td>2017-11-29 16:03:31.497</td>
      <td>Conditions For AI Success: Discipline, Data, A...</td>
      <td>urn:newsml:reuters.com:20171129:nNRA4z4vjf:1</td>
      <td>NS:ABVLAW</td>
      <td>0.089723</td>
      <td>0.406275</td>
      <td>positive</td>
      <td>-0.019657</td>
      <td>-0.058970</td>
      <td>-0.039313</td>
      <td>0.072074</td>
    </tr>
    <tr>
      <th>2017-11-29 13:52:12.000</th>
      <td>2017-11-29 13:52:12.000</td>
      <td>UPDATE 1-Munich Re's Ergo drops plan to sell r...</td>
      <td>urn:newsml:reuters.com:20171129:nL8N1NZ384:2</td>
      <td>NS:RTRS</td>
      <td>0.051498</td>
      <td>0.315673</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-29 11:36:34.130</th>
      <td>2017-11-29 11:36:34.130</td>
      <td>INTERVIEW: IBM advises SEE companies how to pr...</td>
      <td>urn:newsml:reuters.com:20171129:nSEEJkW0na:1</td>
      <td>NS:SEE</td>
      <td>0.134661</td>
      <td>0.383414</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-29 09:45:22.906</th>
      <td>2017-11-29 09:45:22.906</td>
      <td>Grammys 2018: Lady Gaga 'grateful' for two nom...</td>
      <td>urn:newsml:reuters.com:20171129:nNRA4z27k4:1</td>
      <td>NS:ASNEWS</td>
      <td>0.040990</td>
      <td>0.510119</td>
      <td>neutral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-29 00:45:51.955</th>
      <td>2017-11-29 00:45:51.955</td>
      <td>P/f S.s. Railing To Staircase Along With Balan...</td>
      <td>urn:newsml:reuters.com:20171129:nNRA4yy9zn:1</td>
      <td>NS:ECLTND</td>
      <td>0.000000</td>
      <td>0.158333</td>
      <td>neutral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-28 23:46:21.362</th>
      <td>2017-11-28 23:46:21.362</td>
      <td>Annual Repair &amp; Maintenance Operation To Non R...</td>
      <td>urn:newsml:reuters.com:20171128:nNRA4yy0l5:1</td>
      <td>NS:ECLTND</td>
      <td>0.000000</td>
      <td>0.188889</td>
      <td>neutral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-28 23:45:14.000</th>
      <td>2017-11-28 23:45:14.000</td>
      <td>DJ IBM's (IBM) Management Presents at 21st Ann...</td>
      <td>urn:newsml:reuters.com:20171128:nDJW1020F2:1</td>
      <td>NS:DJN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-28 20:45:06.668</th>
      <td>2017-11-28 20:45:06.668</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 205700.0 SHARES O...</td>
      <td>urn:newsml:reuters.com:20171128:nZHN0B647K:1</td>
      <td>NS:RTRS</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>-0.013100</td>
      <td>-0.045851</td>
      <td>-0.104801</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-11-20 21:43:00.418</th>
      <td>2017-11-20 21:43:00.418</td>
      <td>Reuters Insider - Trading at Noon: Another chi...</td>
      <td>urn:newsml:reuters.com:20171120:nRTV1Tl9t2:1</td>
      <td>NS:RTRS</td>
      <td>0.125000</td>
      <td>0.562500</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-20 21:16:47.576</th>
      <td>2017-11-20 21:16:47.576</td>
      <td>Reuters Insider - Is the bottom in for IBM &amp; G...</td>
      <td>urn:newsml:reuters.com:20171120:nRTVbKpGhB:1</td>
      <td>NS:CNBC</td>
      <td>-0.191667</td>
      <td>0.533333</td>
      <td>negative</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-20 19:50:58.000</th>
      <td>2017-11-20 19:51:58.000</td>
      <td>Update: Warren Buffett Buys Apple And Sells IB...</td>
      <td>urn:newsml:reuters.com:20171120:nDJWT016BF:2</td>
      <td>NS:DJN</td>
      <td>0.172727</td>
      <td>0.571591</td>
      <td>positive</td>
      <td>0.046318</td>
      <td>-0.006617</td>
      <td>-0.013234</td>
      <td>-0.350691</td>
    </tr>
    <tr>
      <th>2017-11-20 15:15:25.490</th>
      <td>2017-11-20 15:15:25.490</td>
      <td>Reuters Insider - IBM leads Dow at the open</td>
      <td>urn:newsml:reuters.com:20171120:nRTV1SsXJy:1</td>
      <td>NS:CNBC</td>
      <td>0.050000</td>
      <td>0.200000</td>
      <td>positive</td>
      <td>0.039601</td>
      <td>-0.026401</td>
      <td>0.231008</td>
      <td>0.059402</td>
    </tr>
    <tr>
      <th>2017-11-20 13:09:33.848</th>
      <td>2017-11-20 13:09:33.848</td>
      <td>Deutsche Bank partners with IBM for block-chai...</td>
      <td>urn:newsml:reuters.com:20171120:nNRA4x6kch:1</td>
      <td>NS:ENPNWS</td>
      <td>0.110985</td>
      <td>0.424116</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-20 11:43:00.045</th>
      <td>2017-11-20 11:43:00.045</td>
      <td>Reuters Insider - America has an infertility p...</td>
      <td>urn:newsml:reuters.com:20171120:nRTVc57Gfc:1</td>
      <td>NS:CNBC</td>
      <td>-0.077778</td>
      <td>0.194444</td>
      <td>negative</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-20 08:45:13.875</th>
      <td>2017-11-20 08:45:13.875</td>
      <td>Soundbites: Dubai's Anita Williams back with n...</td>
      <td>urn:newsml:reuters.com:20171120:nNRA4x4jtl:1</td>
      <td>NS:GULNEW</td>
      <td>0.196711</td>
      <td>0.454477</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-19 21:22:30.000</th>
      <td>2017-11-19 21:22:30.000</td>
      <td>IBM could be set for gains after long slump -B...</td>
      <td>urn:newsml:reuters.com:20171119:nL1N1NP0L8:4</td>
      <td>NS:RTRS</td>
      <td>0.082986</td>
      <td>0.196181</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-18 23:52:55.324</th>
      <td>2017-11-18 23:52:55.324</td>
      <td>Comprehensive Amc Of Ibm Blade Center S Server</td>
      <td>urn:newsml:reuters.com:20171118:nNRA4wxol3:1</td>
      <td>NS:ECLTND</td>
      <td>-0.100000</td>
      <td>0.100000</td>
      <td>negative</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-18 11:00:34.000</th>
      <td>2017-11-18 11:00:34.000</td>
      <td>DJ IBM: Blue Chip at a Bargain Price -- Barron's</td>
      <td>urn:newsml:reuters.com:20171118:nDJWR0010D:1</td>
      <td>NS:DJN</td>
      <td>0.111388</td>
      <td>0.402399</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-18 11:00:34.000</th>
      <td>2017-11-18 11:00:34.000</td>
      <td>DJ IBM: Blue Chip at a Bargain Price</td>
      <td>urn:newsml:reuters.com:20171118:nDJWR0010C:1</td>
      <td>NS:DJN</td>
      <td>0.111388</td>
      <td>0.402399</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-18 05:04:55.000</th>
      <td>2017-11-18 05:04:55.000</td>
      <td>DJ IBM: Bargain Blue Chip -- Barrons.com</td>
      <td>urn:newsml:reuters.com:20171118:nDJWR00003:1</td>
      <td>NS:DJN</td>
      <td>0.111388</td>
      <td>0.402399</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-18 04:50:21.836</th>
      <td>2017-11-18 04:50:21.836</td>
      <td>Supply And Delivery Of Ibm Spss Statistics Pre...</td>
      <td>urn:newsml:reuters.com:20171118:nNRA4wrzc7:1</td>
      <td>NS:ECLCTA</td>
      <td>-0.071429</td>
      <td>0.214286</td>
      <td>negative</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-17 21:52:47.180</th>
      <td>2017-11-17 21:52:47.180</td>
      <td>Addressing a cybersecurity skills shortage wit...</td>
      <td>urn:newsml:reuters.com:20171117:nNRA4wljue:1</td>
      <td>NS:GLOBML</td>
      <td>0.160388</td>
      <td>0.463328</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-17 20:45:05.071</th>
      <td>2017-11-17 20:45:05.071</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 138500.0 SHARES O...</td>
      <td>urn:newsml:reuters.com:20171117:nZHN0B5T92:1</td>
      <td>NS:RTRS</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>-0.080440</td>
      <td>-0.060330</td>
      <td>0.033517</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-17 18:28:17.260</th>
      <td>2017-11-17 18:28:17.260</td>
      <td>IBM ST: eye 136.2</td>
      <td>urn:newsml:reuters.com:20171117:nGUR5sxLFT:1</td>
      <td>NS:GURU</td>
      <td>0.005921</td>
      <td>0.325057</td>
      <td>neutral</td>
      <td>0.013379</td>
      <td>0.016724</td>
      <td>-0.040136</td>
      <td>-0.046826</td>
    </tr>
    <tr>
      <th>2017-11-17 16:23:20.614</th>
      <td>2017-11-17 16:23:20.614</td>
      <td>Reuters Insider - Watch one man completely cra...</td>
      <td>urn:newsml:reuters.com:20171117:nRTV9j9KwX:1</td>
      <td>NS:CNBC</td>
      <td>0.066667</td>
      <td>0.366667</td>
      <td>positive</td>
      <td>NaN</td>
      <td>0.006690</td>
      <td>-0.050177</td>
      <td>0.080284</td>
    </tr>
    <tr>
      <th>2017-11-17 05:34:49.483</th>
      <td>2017-11-17 05:34:49.483</td>
      <td>Maintenance Service Of The Certified Datebase ...</td>
      <td>urn:newsml:reuters.com:20171117:nNRA4wib88:1</td>
      <td>NS:ECLCTA</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-17 00:48:31.862</th>
      <td>2017-11-17 00:48:31.862</td>
      <td>3 Factors that Defend IBM ETFs From Berkshire'...</td>
      <td>urn:newsml:reuters.com:20171117:nNRA4wgw0i:1</td>
      <td>NS:ZACKSC</td>
      <td>0.135930</td>
      <td>0.462621</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-17 00:31:27.680</th>
      <td>2017-11-17 00:31:27.680</td>
      <td>Reuters Insider - Cramer Remix: Warren Buffett...</td>
      <td>urn:newsml:reuters.com:20171117:nRTV1kKKFV:1</td>
      <td>NS:CNBC</td>
      <td>0.100000</td>
      <td>0.300000</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-17 00:09:29.826</th>
      <td>2017-11-17 00:09:29.826</td>
      <td>Reuters Insider - Cramer: Thank Wal-Mart and C...</td>
      <td>urn:newsml:reuters.com:20171117:nRTV4Ntxsn:1</td>
      <td>NS:CNBC</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>neutral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-16 15:44:51.000</th>
      <td>2017-11-16 22:39:39.000</td>
      <td>UPDATE 2-IBM urged to avoid working on 'extrem...</td>
      <td>urn:newsml:reuters.com:20171116:nL1N1NM10X:2</td>
      <td>NS:RTRS</td>
      <td>-0.014847</td>
      <td>0.479955</td>
      <td>neutral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-16 20:45:03.579</th>
      <td>2017-11-16 20:45:03.579</td>
      <td>NYSE ORDER IMBALANCE &lt;IBM.N&gt; 124000.0 SHARES O...</td>
      <td>urn:newsml:reuters.com:20171116:nZHN0B5S4D:1</td>
      <td>NS:RTRS</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>-0.013409</td>
      <td>-0.060342</td>
      <td>-0.060342</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-16 17:54:58.928</th>
      <td>2017-11-16 17:54:58.928</td>
      <td>INTERNATIONAL BUSINESS MACHINES CORP SEC Filin...</td>
      <td>urn:newsml:reuters.com:20171116:nEOL641hhv:1</td>
      <td>NS:EDG</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>0.013423</td>
      <td>0.093960</td>
      <td>0.187919</td>
      <td>0.369128</td>
    </tr>
    <tr>
      <th>2017-11-16 17:54:28.684</th>
      <td>2017-11-16 17:54:28.684</td>
      <td>INTERNATIONAL BUSINESS MACHINES CORP SEC Filin...</td>
      <td>urn:newsml:reuters.com:20171116:nEOL1vVXRc:1</td>
      <td>NS:EDG</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>neutral</td>
      <td>0.013423</td>
      <td>0.093960</td>
      <td>0.187919</td>
      <td>0.369128</td>
    </tr>
    <tr>
      <th>2017-11-16 15:00:00.150</th>
      <td>2017-11-16 15:00:00.150</td>
      <td>Optoro Welcomes Jim Kelly as New EVP, Business...</td>
      <td>urn:newsml:reuters.com:20171116:nMKW198mFa:1</td>
      <td>NS:MKW</td>
      <td>0.306273</td>
      <td>0.415809</td>
      <td>positive</td>
      <td>-0.047227</td>
      <td>-0.074214</td>
      <td>-0.033734</td>
      <td>0.033734</td>
    </tr>
    <tr>
      <th>2017-11-16 13:30:54.000</th>
      <td>2017-11-16 13:39:57.000</td>
      <td>Press Release: Fusion Genomics Turns to IBM Cl...</td>
      <td>urn:newsml:reuters.com:20171116:nDJWP00E68:2</td>
      <td>NS:DJN</td>
      <td>0.141170</td>
      <td>0.469894</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-16 13:30:53.372</th>
      <td>2017-11-16 13:30:53.372</td>
      <td>Fusion Genomics Turns to IBM Cloud to Help Sup...</td>
      <td>urn:newsml:reuters.com:20171116:nCNWh74gca:1</td>
      <td>NS:CNW</td>
      <td>0.187248</td>
      <td>0.475294</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-16 13:30:00.000</th>
      <td>2017-11-16 13:30:00.000</td>
      <td>Rights groups pressure IBM to renounce interes...</td>
      <td>urn:newsml:reuters.com:20171116:nL1N1NL22J:2</td>
      <td>NS:RTRS</td>
      <td>0.001837</td>
      <td>0.498600</td>
      <td>neutral</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-11-16 13:00:53.294</th>
      <td>2017-11-16 13:00:53.294</td>
      <td>BMC Mainframe Solutions Accelerate Secure Digi...</td>
      <td>urn:newsml:reuters.com:20171116:nPnKbsbva:1</td>
      <td>NS:PRN</td>
      <td>0.180128</td>
      <td>0.486111</td>
      <td>positive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 11 columns</p>
</div>



Fantastic - we have now completed the analytical part of our study. Finally, we just need to aggregate our results by *Score* bucket in order to draw some conclusions. 


```python
grouped = df.groupby(['Score']).mean()
grouped
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Polarity</th>
      <th>Subjectivity</th>
      <th>twoM</th>
      <th>fiveM</th>
      <th>tenM</th>
      <th>thirtyM</th>
    </tr>
    <tr>
      <th>Score</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>negative</th>
      <td>-0.146508</td>
      <td>0.316746</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>neutral</th>
      <td>0.006436</td>
      <td>0.175766</td>
      <td>-0.004829</td>
      <td>-0.009502</td>
      <td>0.028979</td>
      <td>0.137544</td>
    </tr>
    <tr>
      <th>positive</th>
      <td>0.129260</td>
      <td>0.406868</td>
      <td>0.012089</td>
      <td>0.012776</td>
      <td>0.035936</td>
      <td>0.047345</td>
    </tr>
  </tbody>
</table>
</div>



### Observations

From our initial results - it would appear that there might be some small directional differences in returns between the positive and neutral groups over shorter time frames (twoM and fiveM) after news broke. This is a pretty good basis for further investigation. So where could we go from here?

We have a relatively small *n* here so we might want to increase the size of the study. 

We might also want to try to seperate out more positive or negative news - ie change the threshold of the buckets to try to identify more prominent sentiment articles - maybe that could have more of an impact on performance. 

In terms of capturing news impact - we have thrown a lot of news articles out as they happened outside of market hours - as it is more complex to ascertain impact - we might try to find a way of including some of this in our analysis - I mentioned looking at overseas listings **GDR/ADRs** or surrogates above. Alternatively, we could using **EXACTLY** the same process looking at all news for an index future - say the **S&P500 emini** - as this trades on Globex pretty much round the clock - so we would be throwing out a lot less of the news articles? Great I hear you cry - but would each news article be able to influence a whole index? Are index futures more sensitive to some types of articles than others? Is there a temporal element to this? These are all excellent questions. Or what about cryptocrurrencies? They trade 24/7? and so on.

We could also investigate what is going on with our sentiment engine. We might be able to generate more meaningful results by tinkering with the underlyng processes and parameters. Using a different, more domain-specific corpora might help us to generate more relevant scores. 

You will see there is plenty of scope to get much more involved here. 

This article was intended as an introduction to this most interesting of areas. I hope to have de-mystified this area for you somewhat and shown how it is possible to get started with this type of complex analysis using only a few lines of code, a simple easy to use API and some really fantastic packages, to generate some meaningful results.
