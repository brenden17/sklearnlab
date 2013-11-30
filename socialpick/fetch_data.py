# -*- coding: utf-8 -*-
"""
http://dna.daum.net/apis/socialpick/ref#search
"""
import urllib
import urllib2
import simplejson

SEARCH_BASE = 'http://apis.daum.net/socialpick/search'
args = {'n': '200',
        'output': 'json'}
url = SEARCH_BASE + '?' + urllib.urlencode(args)
count = 0
try:
    response = urllib2.urlopen(url)
    socialdata = simplejson.load(response)
    with open('rawsocialpick.csv', 'a') as f:
        for item in socialdata['socialpick']['item']:
            s = '%s\t%s\t%s\n' % \
                (item['category'], item['keyword'], item['content'])
            f.write(s.encode('utf-8'))
            count += 1
except Exception, e:
    print str(e)
print '%s added' % count
print 'done'