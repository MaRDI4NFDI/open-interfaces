(notes-on-dbus)=
# Notes on DBus

DBus is a system for inter-process communication that provides a way
for applications to talk to each other.
Additionally, DBus provides an event system that allows applications
to subscribe to events to correctly react to them.

There are three principal components: bus, service, and interface.

A **bus** is reponsible for routing messages between applications, for service
registration and discovery.
There is system bus and session bus.

A **service** is an application that provides one or more interfaces to other
applications.
Services self-register with the bus to let others know about them.

An **interface** is a collection of methods, signals, and properties
that a service provides.
Methods are functions that the clients of the service can execute,
signals are emitted by the services to notify clients, and properties
are read/write values.

_Objects_ are service instances that include 1 or more interfaces.
Object have unique path that makes them addressable within a bus.

DBus has client-server architecture.
The bus is the server that has the registry of available services and tracks
services and objects.
The clients communicate with the objects via the server.
