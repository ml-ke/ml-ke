# MQTT Topics for MyŠkoda / VW Connected Car

## Broker
- Host: mqtt.messagehub.de
- Port: 8883 (TLS)
- Auth: Uses the same access token as REST API

## Operation Topics (Client -> Broker)
Used for sending commands to the vehicle. The app publishes to these topics.

```
air-conditioning/set-air-conditioning-at-unlock
air-conditioning/set-air-conditioning-seats-heating
air-conditioning/set-air-conditioning-timers
air-conditioning/set-air-conditioning-without-external-power
air-conditioning/set-target-temperature
air-conditioning/start-stop-air-conditioning
air-conditioning/start-stop-window-heating
air-conditioning/windows-heating
auxiliary-heating/start-stop-auxiliary-heating
charging/start-stop-charging
charging/update-battery-support
charging/update-auto-unlock-plug
charging/update-care-mode
charging/update-charge-limit
charging/update-charge-mode
charging/update-charging-profiles
charging/update-charging-current
departure/update-departure-timers
departure/update-minimal-soc
vehicle-access/honk-and-flash
vehicle-access/lock-vehicle
vehicle-services-backup/apply-backup
vehicle-wakeup/wakeup
```

## Service Event Topics (Broker -> Client)
Broker publishes vehicle state changes to these topics.

```
air-conditioning
charging
departure
vehicle-status/access
vehicle-status/lights
vehicle-status/odometer
```

## Vehicle Event Topics (Broker -> Client)
Broker publishes vehicle connection/ignition state changes.

```
vehicle-connection-status-update
vehicle-ignition-status
```

## Account Event Topics

```
account-event/privacy
```

## Configuration

```python
MQTT_KEEPALIVE = 60
MQTT_RECONNECT_DELAY = 5
MQTT_MAX_RECONNECT_DELAY = 120
MQTT_FAST_RETRY = 10
MQTT_CONNECT_TIMEOUT = 600  # 10 minutes
MQTT_MIN_FCM_REFRESH_INTERVAL = 120  # 2 minutes between FCM token refreshes
MQTT_OPERATION_TIMEOUT = 600  # 10 minutes
```
