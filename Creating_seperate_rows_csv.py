# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pymssql
import re
import pandas as pd
#os.chdir('C:\Users\sharmas\Desktop\Baker & Taylor\Files')


# cur.execute('SELECT c.PatronID, o.ISBN FROM CheckoutTransactionArchive c left join OMNI o ON c.ItemID = o.BTKey WHERE cast(ActionStartDate as Date) between \'20170701\' and \'20181231\' GROUP BY c.PatronID, o.ISBN ORDER BY c.PatronID')
# rows = cur.fetchall()
rows = pd.read_csv('H:/consolidate/input_files/patron_isbn.csv')
rows[1]
fur = conn.cursor()
# fur.execute('SELECT c.PatronID, count(distinct(o.ISBN)) FROM CheckoutTransactionArchive c left join OMNI o ON c.ItemID = o.BTKey WHERE cast(ActionStartDate as Date) between \'20170701\' and \'20181231\' GROUP BY PatronID ORDER BY PatronID')
# mows = fur.fetchall()
mows = pd.read_csv('H:/consolidate/input_files/patron_isbn_checkout.csv')


print('UUIDs starting:')
with open('UUID_dict_V1.txt','w', encoding="utf-8") as f:
    n = 0
    j = 0
    m = 0
    while j <len(mows):
        i = mows[j]
        id = i[0]
        count = i[1]
        f.write(id + '|')
        t = 0
        while t<count:
            g = rows[m][1]
            f.write(g)
            if(t < (count-1)):
                f.write(',')
            else:
                f.write('\n')
            m = m+1
            t = t+1
        if n%10000 == 0:
            print(n,' is finished') 
        n = n + 1
        j = j+1
        
        
#Getting the Data for checkouts


rows = pd.read_csv('H:/consolidate/input_files/isbn_patron.csv')
rows[1]

mows = pd.read_csv('H:/consolidate/input_files/isbn_patron_checkout.csv')

print('checkouts starting:')
with open('Checkouts_dict_V1.txt','w', encoding="utf-8") as f:
    n = 0
    j = 0
    m = 0
    while j <len(mows):
        i = mows[j]
        id = i[0]
        count = i[1]
        f.write(id + '|')
        t = 0
        while t<count:
            g = rows[m][1]
            f.write(g)
            if(t < (count-1)):
                f.write(',')
            else:
                f.write('\n')
            m = m+1
            t = t+1
        if n%10000 == 0:
            print(n,' is finished') 
        n = n + 1
        j = j+1
