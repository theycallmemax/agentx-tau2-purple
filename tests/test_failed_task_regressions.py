from agent import Agent, RESPOND_ACTION_NAME


def test_task14_balance_request_uses_profile_balances():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"user_id":"mohamed_silva_9265","payment_methods":{"gift_card_8020792":{"id":"gift_card_8020792","amount":198},"certificate_3765853":{"id":"certificate_3765853","amount":500},"credit_card_2198526":{"id":"credit_card_2198526"}}}'
    )
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "What are the balances on my gift cards and certificates?"',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "gift_card_8020792" in action["arguments"]["content"]
    assert "certificate_3765853" in action["arguments"]["content"]
    assert "credit_card_2198526" not in action["arguments"]["content"]


def test_task20_selected_flight_number_is_not_treated_as_reservation_id():
    agent = Agent()
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I would like to book flight HAT083."',
        }
    )

    action = agent._guard_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "HAT083"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "flight number" in action["arguments"]["content"]


def test_task22_can_reuse_reservation_inventory_for_other_flight_lookup():
    agent = Agent()
    agent.session_state["reservation_id"] = "D1EW9B"
    agent.session_state["reservation_inventory"] = {
        "D1EW9B": {
            "origin": "ATL",
            "destination": "JFK",
            "dates": ["2024-05-17"],
            "created_at": "2024-05-04T07:38:29",
            "status": None,
            "passenger_count": 1,
        },
        "9HBUV8": {
            "origin": "DEN",
            "destination": "BOS",
            "dates": ["2024-05-17"],
            "created_at": "2024-05-12T17:08:42",
            "status": None,
            "passenger_count": 1,
        },
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Please check my other flight on 2024-05-17."',
        }
    )

    action = agent._guard_action(
        {"name": "get_reservation_details", "arguments": {}}
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "9HBUV8"},
    }


def test_task39_opening_turn_cancel_all_upcoming_starts_with_profile_lookup():
    agent = Agent()
    agent.turn_count = 1
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "My user id is amelia_davis_8890. Yes, I want to cancel all my upcoming flights."',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action == {
        "name": "get_user_details",
        "arguments": {"user_id": "amelia_davis_8890"},
    }


def test_task43_successful_mutation_adds_post_tool_summary_hint():
    agent = Agent()
    agent.session_state["last_tool_name"] = "cancel_reservation"
    agent.messages.append(
        {
            "role": "user",
            "content": 'tool: {"reservation_id":"9HBUV8","status":"cancelled"}',
        }
    )

    messages = agent._messages_for_model()
    joined = "\n".join(
        msg["content"] for msg in messages if isinstance(msg.get("content"), str)
    )

    assert "previous tool call succeeded" in joined.lower()


def test_task44_balance_intent_with_user_id_prefers_lookup_over_refusal():
    agent = Agent()
    agent.turn_count = 1
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "My user ID is sophia_silva_7557. Can you tell me my certificate balances?"',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action == {
        "name": "get_user_details",
        "arguments": {"user_id": "sophia_silva_7557"},
    }


def test_current_run_modify_first_turn_requests_identity_before_search():
    agent = Agent()
    agent.turn_count = 1
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Hi! I need assistance with modifying my upcoming flight. I have a flight from IAH to SEA on May 23, and I would like to push it back to May 24. Also, I want to upgrade the class to business for all passengers. Can you help with that?"',
        }
    )

    action = agent._guard_action(
        {
            "name": "search_direct_flight",
            "arguments": {"origin": "IAH", "destination": "SEA", "date": "2026-05-24"},
        }
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "reservation number or user id" in action["arguments"]["content"].lower()


def test_current_run_remove_passenger_first_turn_requests_identity_before_search():
    agent = Agent()
    agent.turn_count = 1
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Hello! I need to remove a passenger named Sophia from my upcoming round trip flights from LAS to DEN, departing on May 19 and returning on May 20. Can you assist me with that?"',
        }
    )

    action = agent._guard_action(
        {
            "name": "search_direct_flight",
            "arguments": {"origin": "LAS", "destination": "DEN", "date": "2026-05-19"},
        }
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "reservation number or user id" in action["arguments"]["content"].lower()


def test_insurance_question_blocks_destructive_actions():
    agent = Agent()
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Can you check whether insurance is added to reservation PEP4E0?"',
        }
    )

    cancel_action = agent._guard_action(
        {"name": "cancel_reservation", "arguments": {"reservation_id": "PEP4E0"}}
    )
    update_action = agent._guard_action(
        {
            "name": "update_reservation_flights",
            "arguments": {
                "reservation_id": "PEP4E0",
                "cabin": "business",
                "flights": [{"flight_number": "HAT001", "date": "2024-05-19"}],
                "payment_id": "credit_card_1234567",
            },
        }
    )

    assert cancel_action["name"] == RESPOND_ACTION_NAME
    assert update_action["name"] == RESPOND_ACTION_NAME


def test_duplicate_reservation_lookup_is_blocked_for_same_turn():
    agent = Agent()
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Please check reservation 3JA7XV."',
        }
    )
    agent.session_state["last_tool_name"] = "get_reservation_details"
    agent.session_state["last_tool_arguments"] = {"reservation_id": "3JA7XV"}
    agent.session_state["last_tool_user_text"] = "Please check reservation 3JA7XV."
    agent.session_state["last_tool_streak"] = 2

    action = agent._guard_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "3JA7XV"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "already checked" in action["arguments"]["content"].lower()


def test_cancel_is_blocked_when_policy_makes_basic_economy_ineligible():
    agent = Agent()
    agent.messages.append(
        {
            "role": "user",
            "content": 'tool: {"reservation_id":"EHGLP3","user_id":"emma_kim_9957","origin":"PHX","destination":"JFK","flight_type":"one_way","cabin":"basic_economy","flights":[{"flight_number":"HAT156","origin":"PHX","destination":"SEA","date":"2024-05-17","price":50},{"flight_number":"HAT021","origin":"SEA","destination":"JFK","date":"2024-05-17","price":54}],"passengers":[{"first_name":"Evelyn","last_name":"Taylor","dob":"1965-01-16"},{"first_name":"Anya","last_name":"Silva","dob":"1971-11-22"}],"payment_history":[{"payment_id":"credit_card_5832574","amount":208}],"created_at":"2024-05-04T23:12:06","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}',
        }
    )
    agent._update_state_from_tool_payload(agent.messages[-1]["content"])
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Yes, my user ID is emma_kim_9957. The reason for cancellation is a change of plans. Please go ahead and cancel the reservation."',
        }
    )
    agent.session_state["pending_confirmation_action"] = "cancel_reservation"

    action = agent._guard_action(
        {"name": "cancel_reservation", "arguments": {"reservation_id": "EHGLP3"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "not eligible for cancellation" in action["arguments"]["content"].lower()


def test_cabin_only_change_confirmation_is_deterministic():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"GV1N64","user_id":"james_patel_9828","origin":"LAS","destination":"DEN","flight_type":"round_trip","cabin":"business","flights":[{"flight_number":"HAT003","origin":"LAS","destination":"DEN","date":"2024-05-19","price":561},{"flight_number":"HAT290","origin":"DEN","destination":"LAS","date":"2024-05-20","price":1339}],"payment_history":[{"payment_id":"gift_card_1642017","amount":5700}],"created_at":"2024-05-03T05:35:00","total_baggages":3,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Please downgrade reservation GV1N64 to basic economy and let me know the refund."',
        }
    )
    agent._update_state_from_text(agent.messages[-1]["content"])

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "change reservation gv1n64 from business to basic_economy" in action["arguments"]["content"].lower()


def test_confirmed_cabin_only_change_uses_same_flights_update():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"GV1N64","user_id":"james_patel_9828","origin":"LAS","destination":"DEN","flight_type":"round_trip","cabin":"business","flights":[{"flight_number":"HAT003","origin":"LAS","destination":"DEN","date":"2024-05-19","price":561},{"flight_number":"HAT290","origin":"DEN","destination":"LAS","date":"2024-05-20","price":1339}],"payment_history":[{"payment_id":"gift_card_1642017","amount":5700}],"created_at":"2024-05-03T05:35:00","total_baggages":3,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.session_state["requested_cabin"] = "basic_economy"
    agent.session_state["cabin_only_change"] = True
    agent.session_state["pending_confirmation_action"] = "update_reservation_flights"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Yes, please proceed with the downgrade."',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action == {
        "name": "update_reservation_flights",
        "arguments": {
            "reservation_id": "GV1N64",
            "cabin": "basic_economy",
            "flights": [
                {"flight_number": "HAT003", "date": "2024-05-19"},
                {"flight_number": "HAT290", "date": "2024-05-20"},
            ],
            "payment_id": "gift_card_1642017",
        },
    }


def test_cabin_only_modify_blocks_baggage_tool_misroute():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"YAX4DR","user_id":"chen_lee_6825","origin":"BOS","destination":"MSP","flight_type":"one_way","cabin":"economy","flights":[{"flight_number":"HAT235","origin":"BOS","destination":"MCO","date":"2024-05-18","price":122},{"flight_number":"HAT298","origin":"MCO","destination":"MSP","date":"2024-05-19","price":127}],"payment_history":[{"payment_id":"credit_card_4938634","amount":498}],"created_at":"2024-05-05T23:00:15","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.session_state["requested_cabin"] = "business"
    agent.session_state["cabin_only_change"] = True
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Yes, please upgrade the reservation to business."',
        }
    )

    action = agent._guard_action(
        {
            "name": "update_reservation_baggages",
            "arguments": {
                "reservation_id": "YAX4DR",
                "total_baggages": 2,
                "nonfree_baggages": 0,
                "payment_id": "credit_card_4938634",
            },
        }
    )

    assert action == {
        "name": "update_reservation_flights",
        "arguments": {
            "reservation_id": "YAX4DR",
            "cabin": "business",
            "flights": [
                {"flight_number": "HAT235", "date": "2024-05-18"},
                {"flight_number": "HAT298", "date": "2024-05-19"},
            ],
            "payment_id": "credit_card_4938634",
        },
    }


def test_baggage_update_is_normalized_to_free_allowance():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"user_id":"chen_lee_6825","membership":"gold","payment_methods":{"credit_card_4938634":{"id":"credit_card_4938634"}}}'
    )
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"YAX4DR","user_id":"chen_lee_6825","origin":"BOS","destination":"MSP","flight_type":"one_way","cabin":"economy","flights":[{"flight_number":"HAT235","origin":"BOS","destination":"MCO","date":"2024-05-18","price":122},{"flight_number":"HAT298","origin":"MCO","destination":"MSP","date":"2024-05-19","price":127}],"passengers":[{"first_name":"Chen","last_name":"Lee","dob":"1967-12-12"},{"first_name":"Noah","last_name":"Hernandez","dob":"1968-01-06"}],"payment_history":[{"payment_id":"credit_card_4938634","amount":498}],"created_at":"2024-05-05T23:00:15","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Please add 2 checked bags to reservation YAX4DR."',
        }
    )

    action = agent._guard_action(
        {
            "name": "update_reservation_baggages",
            "arguments": {
                "reservation_id": "YAX4DR",
                "total_baggages": 2,
                "nonfree_baggages": 2,
                "payment_id": "credit_card_4938634",
            },
        }
    )

    assert action == {
        "name": "update_reservation_baggages",
        "arguments": {
            "reservation_id": "YAX4DR",
            "total_baggages": 2,
            "nonfree_baggages": 0,
            "payment_id": "credit_card_4938634",
        },
    }


def test_upgrade_cost_request_forces_calculate_tool():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"YAX4DR","user_id":"chen_lee_6825","origin":"BOS","destination":"MSP","flight_type":"one_way","cabin":"economy","flights":[{"flight_number":"HAT235","origin":"BOS","destination":"MCO","date":"2024-05-18","price":122},{"flight_number":"HAT298","origin":"MCO","destination":"MSP","date":"2024-05-19","price":127}],"passengers":[{"first_name":"Chen","last_name":"Lee","dob":"1967-12-12"},{"first_name":"Noah","last_name":"Hernandez","dob":"1968-01-06"}],"payment_history":[{"payment_id":"credit_card_4938634","amount":498}],"created_at":"2024-05-05T23:00:15","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.session_state["requested_cabin"] = "business"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Please check the total price for upgrading both passengers to business. I only want it if it is under $650."',
        }
    )
    agent.session_state["last_tool_name"] = "search_direct_flight"
    agent.session_state["last_tool_arguments"] = {
        "origin": "BOS",
        "destination": "MCO",
        "date": "2024-05-18",
    }
    agent._update_state_from_tool_payload(
        'tool: [{"flight_number":"HAT235","origin":"BOS","destination":"MCO","prices":{"business":350}}]'
    )
    agent.session_state["last_tool_arguments"] = {
        "origin": "MCO",
        "destination": "MSP",
        "date": "2024-05-19",
    }
    agent._update_state_from_tool_payload(
        'tool: [{"flight_number":"HAT298","origin":"MCO","destination":"MSP","prices":{"business":499}}]'
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action == {
        "name": "calculate",
        "arguments": {"expression": "2 * ((350 - 122) + (499 - 127))"},
    }


def test_affirmation_with_cost_constraint_prefers_pricing_over_write():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"YAX4DR","user_id":"chen_lee_6825","origin":"BOS","destination":"MSP","flight_type":"one_way","cabin":"economy","flights":[{"flight_number":"HAT235","origin":"BOS","destination":"MCO","date":"2024-05-18","price":122},{"flight_number":"HAT298","origin":"MCO","destination":"MSP","date":"2024-05-19","price":127}],"passengers":[{"first_name":"Chen","last_name":"Lee","dob":"1967-12-12"},{"first_name":"Noah","last_name":"Hernandez","dob":"1968-01-06"}],"payment_history":[{"payment_id":"credit_card_4938634","amount":498}],"created_at":"2024-05-05T23:00:15","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.session_state["requested_cabin"] = "business"
    agent.session_state["cabin_only_change"] = True
    agent.session_state["pending_confirmation_action"] = "update_reservation_flights"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Yes, please go ahead and check the cost for upgrading both passengers to business class. If it is over $650, then just upgrade Noah."',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action == {
        "name": "search_direct_flight",
        "arguments": {
            "origin": "BOS",
            "destination": "MCO",
            "date": "2024-05-18",
        },
    }


def test_payment_id_is_not_misparsed_as_user_id():
    agent = Agent()
    agent.session_state["user_id"] = "chen_lee_6825"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Yes, please use credit_card_4938634 for the checked bags."',
        }
    )

    action = agent._guard_action(
        {"name": "get_user_details", "arguments": {"user_id": "credit_card_4938634"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "user id or reservation number" in action["arguments"]["content"].lower()


def test_modify_future_reservation_blocks_flight_status_tool():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"YAX4DR","user_id":"chen_lee_6825","origin":"BOS","destination":"MSP","flight_type":"one_way","cabin":"economy","flights":[{"flight_number":"HAT235","origin":"BOS","destination":"MCO","date":"2024-05-18","price":122},{"flight_number":"HAT298","origin":"MCO","destination":"MSP","date":"2024-05-19","price":127}],"passengers":[{"first_name":"Chen","last_name":"Lee","dob":"1967-12-12"},{"first_name":"Noah","last_name":"Hernandez","dob":"1968-01-06"}],"payment_history":[{"payment_id":"credit_card_4938634","amount":498}],"created_at":"2024-05-05T23:00:15","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.session_state["requested_cabin"] = "business"
    agent.session_state["task_type"] = "modify"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Please verify the flights have not been flown yet before changing the cabin."',
        }
    )

    action = agent._guard_action(
        {
            "name": "get_flight_status",
            "arguments": {"flight_number": "HAT235", "date": "2024-05-18"},
        }
    )

    assert action == {
        "name": "search_direct_flight",
        "arguments": {
            "origin": "BOS",
            "destination": "MCO",
            "date": "2024-05-18",
        },
    }


def test_pricing_expression_uses_baseline_reservation_prices():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"YAX4DR","user_id":"chen_lee_6825","origin":"BOS","destination":"MSP","flight_type":"one_way","cabin":"economy","flights":[{"flight_number":"HAT235","origin":"BOS","destination":"MCO","date":"2024-05-18","price":122},{"flight_number":"HAT298","origin":"MCO","destination":"MSP","date":"2024-05-19","price":127}],"passengers":[{"first_name":"Chen","last_name":"Lee","dob":"1967-12-12"},{"first_name":"Noah","last_name":"Hernandez","dob":"1968-01-06"}],"payment_history":[{"payment_id":"credit_card_4938634","amount":498}],"created_at":"2024-05-05T23:00:15","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"YAX4DR","user_id":"chen_lee_6825","origin":"BOS","destination":"MSP","flight_type":"one_way","cabin":"business","flights":[{"flight_number":"HAT235","origin":"BOS","destination":"MCO","date":"2024-05-18","price":350},{"flight_number":"HAT298","origin":"MCO","destination":"MSP","date":"2024-05-19","price":499}],"passengers":[{"first_name":"Chen","last_name":"Lee","dob":"1967-12-12"},{"first_name":"Noah","last_name":"Hernandez","dob":"1968-01-06"}],"payment_history":[{"payment_id":"credit_card_4938634","amount":498},{"payment_id":"credit_card_4938634","amount":1200}],"created_at":"2024-05-05T23:00:15","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.session_state["flight_search_inventory"] = {
        "BOS|MCO|2024-05-18": [
            {"flight_number": "HAT235", "prices": {"business": 350}},
        ],
        "MCO|MSP|2024-05-19": [
            {"flight_number": "HAT298", "prices": {"business": 499}},
        ],
    }

    assert agent._pricing_expression_for_current_reservation("YAX4DR", "business") == (
        "2 * ((350 - 122) + (499 - 127))"
    )


def test_declining_expensive_upgrade_clears_pending_cabin_intent():
    agent = Agent()
    agent.session_state["requested_cabin"] = "business"
    agent.session_state["cabin_only_change"] = True
    agent.session_state["pending_confirmation_action"] = "update_reservation_flights"

    agent._update_state_from_text(
        "Since the additional cost is over $1000, I can't proceed with that upgrade. Just add the bags instead."
    )

    assert agent.session_state["requested_cabin"] is None
    assert agent.session_state["cabin_only_change"] is False
    assert agent.session_state["pending_confirmation_action"] is None
