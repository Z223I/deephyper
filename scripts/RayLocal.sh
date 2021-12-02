#!/bin/bash

head_node=$HOSTNAME
echo $HOSTNAME
head_node_ip=$(dig $head_node a +short | awk 'FNR==2')
head_node_ip=127.0.0.1
echo ">$head_node_ip<"

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
head_node_ip=${ADDR[1]}
else
head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# Starting the Ray Head Node
# port=6379 original
port=6380
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
echo "ray start --head --node-ip-address=$head_node_ip --port=$port --block &"
ray start --head --node-ip-address=$head_node_ip --port=$port --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10