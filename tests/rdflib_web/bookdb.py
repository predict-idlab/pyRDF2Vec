"""
A tiny test graph with Book meta-data
"""

import rdflib
import io

bookrdf="""
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix book: <http://example.org/book/> .
@prefix dc: <http://purl.org/dc/elements/1.1/> .
@prefix vcard: <http://www.w3.org/2001/vcard-rdf/3.0#> .

<http://example.org/book/book1> a book:Book ;
    dc:creator "J.K. Rowling";
    dc:title "Harry Potter and the Philosopher\'s Stone" .

<http://example.org/book/book2> a book:Book ;
    book:author <http://example.org/book/MrsRowling> ;
    dc:title "Harry Potter & the Chamber of Secrets" .

<http://example.org/book/book3> a book:Book ;
    dc:creator <http://example.org/book/MrsRowling>;
    dc:title "Harry Potter and the Prisoner Of Azkaban" .

<http://example.org/book/book4> dc:title "Harry Potter and the Goblet of Fire" .

<http://example.org/book/book5> a book:Book ;
    dc:creator "J.K. Rowling";
    dc:title "Harry Potter and the Order of the Ph\xc3\xb6nix" .

<http://example.org/book/book6> a book:Book ;
    dc:creator "J.K. Rowling";
    dc:title "Harry Potter and the Half-Blood Prince" .

<http://example.org/book/book7> a book:Book ;
    dc:creator "J.K. Rowling";
    dc:title "Harry Potter and the Deathly Hallows" .

<http://example.org/book/b\xc3\xb6\xc3\xb6k8> a book:Book ;
    dc:creator "Moosy";
    dc:title "Moose bite can b\xc3\xb6 very nasty."@se ;
    dc:title "Elgbitt kan v\xc3\xa6re veldig vondt."@no ; .

<http://example.org/book/MrsRowling> a book:Person ;
    vcard:Family "Rowling";
    vcard:Given "Joanna" .

book:Work a rdfs:Class ;
    rdfs:label "Work" .

book:Publication a rdfs:Class ;
    rdfs:label "Publication" .

book:Book a rdfs:Class ;
    rdfs:subClassOf book:Work, book:Publication ;
    rdfs:label "Book" .

dc:creator a rdf:Property ;
    rdfs:label "creator" ;
    rdfs:domain book:Book .

book:author a rdf:Property ;
    rdfs:label "author" ;
    rdfs:domain book:Book ;
    rdfs:subPropertyOf dc:creator .


"""

bookdb=rdflib.Graph()
bookdb.parse(data=bookrdf,format='n3')
