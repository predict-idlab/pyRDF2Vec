
try: 
    import mimeparse
except: 
    import warnings
    warnings.warn("mimeparse not found - I need this for content negotiation, install with 'easy_install mimeparse'")
    mimeparse=None
    
# sparql results
JSON_MIME="application/sparql-results+json"
XML_MIME="application/sparql-results+xml"

HTML_MIME="text/html"
N3_MIME="text/n3"
TURTLE_MIME="text/turtle"
RDFXML_MIME="application/rdf+xml"
NTRIPLES_MIME="text/plain"
JSONLD_MIME="application/json"

FORMAT_MIMETYPE={ "rdf": RDFXML_MIME, "n3": N3_MIME, "nt": NTRIPLES_MIME, "turtle": TURTLE_MIME, "json-ld": JSONLD_MIME }
MIMETYPE_FORMAT=dict(list(map(reversed,list(FORMAT_MIMETYPE.items()))))

def mime_to_format(mimetype): 
    if mimetype in MIMETYPE_FORMAT:
        return MIMETYPE_FORMAT[mimetype]
    return "rdf"
    
def format_to_mime(format): 
    if format=='ttl': format='turtle'
    if format=='json': format='json-ld'
    if format in FORMAT_MIMETYPE:
        return format, FORMAT_MIMETYPE[format]
    return "xml", RDFXML_MIME
    
    

def resultformat_to_mime(format): 
    if format=='xml': return XML_MIME
    if format=='json': return JSON_MIME
    if format=='html': return HTML_MIME
    return "text/plain"
    
def best_match(cand, header): 
    if mimeparse and header:
        return mimeparse.best_match(cand,header)
    return None
