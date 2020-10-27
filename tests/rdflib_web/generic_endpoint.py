import rdflib

class DefaultGraphReadOnly(Exception):
    pass

class NamedGraphsNotSupported(Exception):
    pass

class GenericEndpoint:
    """
    This is an implementation of the SPAQL 1.1 Protocol and SPARQL 1.1
    Graph Store Protocol suitable for integration into webserver
    frameworks.
    """

    def __init__(self, ds, coin_url):
        """
        :argument:ds: The dataset to be used. Must be a Dataset (recommeded),
        ConjunctiveGraph or Graph. In case of a Graph, it is served as
        the default graph.
        :argument:coin_url: A function that takes no arguments and outputs
        an URI for a fresh graph. This is used when graph_identifier is
        None and neither 'default' nor 'graph' can be found in args.
        """
        self.ds = ds
        self.coin_url = coin_url

    DEFAULT = 'DEFAULT'

    RESULT_GRAPH = 0

    def negotiate(self, resulttype, accept_header):
        #TODO: Find all mimetypes supported by the serializers
        #automatically instead of hardcoded
        import logging
        if resulttype == self.RESULT_GRAPH:
            available = ['application/n-triples', 'text/n3', 'text/turtle', 'application/rdf+xml']
        assert available, "Invalid resulttype"
        from . import mimeutils
        best = mimeutils.best_match(available, accept_header) or available[-1]
        return best, best

    def graph_store(self, method, graph_identifier, args, body, mimetype, accept_header):
        """Handles a request according to the SPARQL 1.1 graph store
        protocol.

        :argument:method: 'PUT', 'POST', 'DELETE', or 'GET'
        :argument:graph_identifier: rdflib.URIRef of the graph against
        which the request is made. It must be None for indirect requests. The
        special value GenericEndpoint.DEFAULT denotes the default graph. 
        :argument:args: A dict containing all URL parameters
        :argument:body: The request body as list of dicts if the
        content-type is multipart/form-data, otherwise a string.
        :argument:mimetype: The mime type part (i.e. without charset) of
        the request body
        :argument:accept_header: The accept header value as given by
        the client. This is required for content negotiation.

        :Returns:

        A triple consisting of the HTTP status code, a dictionary of
        headers that must be included in the status, and the body of
        the status. In case of an error (i.e. the status code is
        at least 400), then the body only consists of a error message
        as string. In this case, the caller is responsible to create a
        proper status body. If the status code is 201 or 204, the body
        is None.

        This method can through exceptions. If this happens, it is always an
        internal error.
        """
        if not graph_identifier:
            if 'default' in args:
                graph_identifier = self.DEFAULT
            elif 'graph' in args:
                graph_identifier = rdflib.URIRef(args['graph'])
            elif method == 'POST':
                graph_identifier = None
            else:
                return (400, dict(), "Missing URL query string parameter 'graph' or 'default'")

        existed = False
        if graph_identifier == self.DEFAULT:
            existed = True
        elif graph_identifier and self.ds.context_aware:
            existed = graph_identifier in {g.identifier for g in self.ds.contexts()}

        def get_graph(identifier):
            # Creates the graph if it does not already exist and returns
            # it.
            if graph_identifier == self.DEFAULT:
                # A ConjunctiveGraph or Datset itself represents the
                # default graph (it might be the union of all graphs).
                # In case of a plain Graph, the default graph is the
                # graph itself too.
                return self.ds
            elif hasattr(self.ds, "graph"): # No Graph.graph_aware
                return self.ds.graph(identifier)
            elif self.ds.context_aware:
                return self.ds.get_context(identifier)
            else:
                raise NamedGraphsNotSupported()

        def clear_graph(identifier):
            if identifier == self.DEFAULT:
                if self.ds.default_union:
                    raise DefaultGraphReadOnly()
                elif self.ds.context_aware:
                    self.ds.default_context.remove((None,None,None))
                else:
                    self.ds.remove((None,None,None))
            else:
                self.ds.remove((None, None, None, get_graph(identifier)))

        def remove_graph(identifier):
            # Unfortunately, there is no Graph.graph_aware, so use
            # hasattr
            if identifier == self.DEFAULT and self.ds.default_union:
                raise DefaultGraphReadOnly()
            elif hasattr(self.ds, "remove_graph"):
                self.ds.remove_graph(get_graph(identifier))
            else:
                clear_graph(identifier)

        def parseInto(target, data, format):
            # Makes shure that the for ConjucntiveGraph and Dataset we
            # parse into the default graph instead of into a fresh
            # graph.
            if target.default_union:
                raise DefaultGraphReadOnly()
            if target.context_aware:
                target.default_context.parse(data=data, format=format)
            else:
                target.parse(data=data, format=format)

        try:

            if method == 'PUT':
                if existed:
                    clear_graph(graph_identifier)
                target = get_graph(graph_identifier)
                parseInto(target, data=body, format=mimetype)
                response = (204 if existed else 201, dict(), None)

            elif method == 'DELETE':
                if existed:
                    remove_graph(graph_identifier)
                    response = (204, dict(), None)
                else:
                    response = (404, dict(), 'Graph %s not found' % graph_identifier)

            elif method == 'POST':
                additional_headers = dict()
                if not graph_identifier:
                    # Coin a new identifier
                    existed = False
                    url = self.coin_url()
                    graph_identifier = rdflib.URIRef(url)
                    additional_headers['location'] = url
                target = get_graph(graph_identifier)
                if mimetype == "multipart/form-data":
                    for post_item in body:
                        target = get_graph(graph_identifier)
                        parseInto(target, data=post_item['data'], format=post_item['mimetype'])
                else:
                    parseInto(target, data=body, format=mimetype)
                response = (204 if existed else 201, additional_headers, None)

            elif method == 'GET' or method == 'HEAD':
                if existed:
                    format, content_type = self.negotiate(self.RESULT_GRAPH, accept_header)
                    if content_type.startswith('text/'): content_type += "; charset=utf-8"
                    headers = {"Content-type": content_type}
                    response = (200, headers, get_graph(graph_identifier).serialize(format=format))
                else:
                    response = (404, dict(), 'Graph %s not found' % graph_identifier)

            else:
                response = (405, {"Allow": "GET, HEAD, POST, PUT, DELETE"}, "Method %s not supported" % method)

        except DefaultGraphReadOnly:
            response = (400, dict(), "Default graph is read only because it is the uion")
        except NamedGraphsNotSupported:
            response = (400, dict(), "Named graphs not supported")

        return response

