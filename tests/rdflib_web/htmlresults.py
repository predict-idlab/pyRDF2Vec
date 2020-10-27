"""

SPARQL Result Serializer and RDF Serializer 
for nice human readable HTML tables

>>> from rdflib import RDF, RDFS, XSD, Graph, Literal
>>> g=Graph()
>>> g.add((RDF.Property, RDF.type, RDFS.Class))
>>> g.add((RDF.Property, RDFS.label, Literal('Property')))
>>> g.add((RDF.Property, RDFS.label, Literal('Property', datatype=XSD.string)))
>>> g.add((RDF.Property, RDFS.label, Literal('Eigenschaft', lang='de')))

>>> s=g.query('select * where { ?s ?p ?o . }').serialize(format='html')
>>> 'rdf:type' in s
True
>>> '@de' in s
True
>>> '^^&lt;xsd:string&gt;' in s
True
"""

import rdflib
from rdflib.query import ResultSerializer
from rdflib.serializer import Serializer

import warnings

from jinja2 import Environment, contextfilter, Markup

nm=rdflib.Graph().namespace_manager
nm.bind('xsd',rdflib.XSD)

def qname(ctx, t):
    try:
        if "graph" in ctx: 
            l=ctx["graph"].namespace_manager.compute_qname(t, False)
        else: 
            l=nm.compute_qname(t, False)
        return '%s:%s'%(l[0],l[2])
    except: 
        return t


@contextfilter
def term_to_string(ctx, t): 
    if isinstance(t, rdflib.URIRef):
        l=qname(ctx,t)
        return Markup("<a href='%s'>%s</a>"%(t,l))
    elif isinstance(t, rdflib.Literal): 
        if t.language: 
            return '"%s"@%s'%(t,t.language)
        elif t.datatype: 
            return '"%s"^^&lt;%s&gt;'%(t,qname(ctx,t.datatype))
        else:
            return '"%s"'%t
    return t

env=Environment()
env.filters["term_to_string"]=term_to_string


GRAPH_TEMPLATE="""
<table>
<thead>
 <tr>
  <th>subject</th>
  <th>predicate</th>
  <th>object</th>
 </tr>
</thead>
<tbody>
 {% for t in graph %}
  <tr>
  {% for x in t %}
   <td>{{x|term_to_string}}</td>
  {% endfor %}
  </tr>
 {% endfor %}
</tbody>
</table>

"""

SELECT_TEMPLATE="""
<table>
<thead>
 <tr>
 {% for var in result.vars %}
  <th>{{var}}</th>
 {% endfor %}
 </tr>
</thead>
<tbody>
 {% for row in result.bindings %}
  <tr>
  {% for var in result.vars %}
   <td>{{row[var]|term_to_string}}</td>
  {% endfor %}
  </tr>
 {% endfor %}
</tbody>
</table>

"""


class HTMLResultSerializer(ResultSerializer):

    def __init__(self, result): 
        ResultSerializer.__init__(self, result)

    def serialize(self, stream, encoding="utf-8"):
        if self.result.type=='ASK':
            stream.write("<strong>true</strong>".encode(encoding))
            return
        if self.result.type=='SELECT':
            template = env.from_string(SELECT_TEMPLATE)
            stream.write(template.render(result=self.result))


            



class HTMLSerializer(Serializer):
    """
    Serializes RDF graphs as HTML tables
    """

    def serialize(self, stream, base=None, encoding=None, **args):
        if base is not None:
            warnings.warn("HTMLSerializer does not support base.")
        if encoding is not None:
            warnings.warn("HTMLSerializer does not use custom encoding.")

        template = env.from_string(GRAPH_TEMPLATE)
        res=template.render(graph=self.store)
        if not encoding: encoding="utf-8"
        
        res=res.encode(encoding)
        stream.write(res)


if __name__=='__main__':
    import rdflib
    g=rdflib.Graph()
    g.add((rdflib.RDF.Property, rdflib.RDF.type, rdflib.RDFS.Class))
    g.add((rdflib.RDF.Property, rdflib.RDFS.label, rdflib.Literal('Property')))
    g.add((rdflib.RDF.Property, rdflib.RDFS.label, rdflib.Literal('Property', datatype=rdflib.XSD.string)))
    g.add((rdflib.RDF.Property, rdflib.RDFS.label, rdflib.Literal('Eigenschaft', lang='de')))

    s=g.query('select * where { ?s ?p ?o . }').serialize(format='html')

