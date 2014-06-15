# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 00:38:37 2014

@author: m
"""
import httplib, urllib
def send_note(msg):
	#msg= "hello world"
	conn = httplib.HTTPSConnection("api.pushover.net:443")
	conn.request("POST", "/1/messages.json",
  	urllib.urlencode({
   	 "token": "acfZ42h7KMGmAdbzyCBZxkDwTrzhPN",
   	 "user": "uxFdSnAMc9D9kcBdgZWYkW3mwynUvc",
   	 "message": msg,
  	}), { "Content-type": "application/x-www-form-urlencoded" })
	conn.getresponse()
#%%
send_note("trial mid")