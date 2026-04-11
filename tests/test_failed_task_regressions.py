from agent import Agent, RESPOND_ACTION_NAME
from intent import classify_task_type
from policy_sanitizer import sanitize_tool_descriptions
from runtime import sync_runtime_state, termination_controller


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
    assert "Total gift cards: $198.00" in action["arguments"]["content"]
    assert "Total certificates: $500.00" in action["arguments"]["content"]


def test_compensation_story_is_classified_as_status_not_cancel():
    text = (
        "I had a business flight earlier this month that was canceled, "
        "and I deserve compensation for the inconvenience and the missed meeting."
    )

    assert classify_task_type(text) == "status"


def test_direct_cancel_request_still_classifies_as_cancel():
    text = "Please cancel my reservation because my earlier flight was canceled."

    assert classify_task_type(text) == "cancel"


def test_baggage_followup_clears_adversarial_compensation_state():
    agent = Agent()
    agent.session_state["working_memory"] = {
        "adversarial_compensation_verification": True,
        "adversarial_compensation_business_claim": True,
        "adversarial_compensation_cancelled_claim": True,
    }

    agent._update_state_from_text(
        "I am not asking about compensation. I just need to know how many suitcases I can bring."
    )

    working = agent.session_state["working_memory"]
    assert working["adversarial_compensation_verification"] is False
    assert working["adversarial_compensation_business_claim"] is False
    assert working["adversarial_compensation_cancelled_claim"] is False
    assert agent.session_state["task_type"] == "baggage"


def test_cancel_flow_after_reservation_lookup_refuses_without_extra_identity_prompt():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["latest_input_is_tool"] = True
    agent.session_state["last_tool_name"] = "get_reservation_details"
    agent.session_state["reservation_id"] = "EHGLP3"
    agent.session_state["loaded_reservation_details"] = True
    agent.session_state["reservation_inventory"] = {
        "EHGLP3": {
            "origin": "PHX",
            "destination": "JFK",
            "created_at": "2024-05-10T11:00:00",
            "status": None,
            "cabin": "basic_economy",
            "insurance": "no",
            "flights": [
                {"flight_number": "HAT156", "origin": "PHX", "destination": "SEA", "date": "2024-05-17"},
                {"flight_number": "HAT021", "origin": "SEA", "destination": "JFK", "date": "2024-05-17"},
            ],
        }
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I would like to cancel this reservation and get a refund."',
        }
    )
    result = agent._graph_llm_decide(
        {"latest_user_text": "I would like to cancel this reservation and get a refund."}
    )

    assert result["action"]["name"] == RESPOND_ACTION_NAME
    assert "not eligible for cancellation" in result["action"]["arguments"]["content"].lower()


def test_identity_amnesia_with_route_context_loads_next_reservation_instead_of_reasking_user():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["loaded_user_details"] = True
    agent.session_state["user_id"] = "raj_sanchez_7340"
    agent.session_state["origin"] = "PHL"
    agent.session_state["destination"] = "LGA"
    agent.session_state["known_reservation_ids"] = ["MZDDS4", "Q69X3R"]
    agent.session_state["reservation_inventory"] = {}
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I want to cancel my trip from Philadelphia to LaGuardia."',
        }
    )

    action = agent._guard_action(
        {
            "name": "respond",
            "arguments": {
                "content": "Please share your reservation number or user ID so I can locate the booking you want to cancel."
            },
        }
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "MZDDS4"},
    }


def test_cancel_post_user_lookup_with_multiple_reservations_asks_for_disambiguation():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["latest_input_is_tool"] = True
    agent.session_state["last_tool_name"] = "get_user_details"
    agent.session_state["loaded_user_details"] = True
    agent.session_state["known_reservation_ids"] = ["MZDDS4", "60RX9E", "Q69X3R"]
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I need help canceling a reservation I have."',
        }
    )

    result = agent._graph_llm_decide(
        {"latest_user_text": "I need help canceling a reservation I have."}
    )

    assert result["action"]["name"] == RESPOND_ACTION_NAME
    assert "multiple reservations" in result["action"]["arguments"]["content"].lower()


def test_forced_cancel_identity_action_still_runs_through_guard():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    result = agent._graph_llm_decide(
        {"latest_user_text": "Please cancel reservation EHGLP3."}
    )
    assert result["action"] == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "EHGLP3"},
    }


def test_aggregate_cancel_post_user_lookup_iterates_all_known_reservations():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["latest_input_is_tool"] = True
    agent.session_state["last_tool_name"] = "get_user_details"
    agent.session_state["loaded_user_details"] = True
    agent.session_state["known_reservation_ids"] = ["8C8K4E", "LU15PA", "MSJ4OA"]
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Please cancel all upcoming flights on my profile."',
        }
    )

    result = agent._graph_llm_decide(
        {"latest_user_text": "Please cancel all upcoming flights on my profile."}
    )

    assert result["action"] == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "8C8K4E"},
    }


def test_cancel_followup_with_reason_refuses_immediately_once_reservation_is_loaded():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["reservation_id"] = "Q69X3R"
    agent.session_state["loaded_reservation_details"] = True
    agent.session_state["reservation_inventory"] = {
        "Q69X3R": {
            "origin": "PHL",
            "destination": "LGA",
            "created_at": "2024-05-14T09:52:38",
            "status": None,
            "cabin": "economy",
            "insurance": "no",
            "flights": [
                {"flight_number": "HAT243", "origin": "PHL", "destination": "CLT", "date": "2024-05-20"},
                {"flight_number": "HAT024", "origin": "CLT", "destination": "LGA", "date": "2024-05-20"},
            ],
        }
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I no longer need to make the trip, so I would like to cancel it."',
        }
    )

    result = agent._graph_llm_decide(
        {"latest_user_text": "I no longer need to make the trip, so I would like to cancel it."}
    )

    assert result["action"]["name"] == RESPOND_ACTION_NAME
    assert "not eligible for cancellation" in result["action"]["arguments"]["content"].lower()


def test_cancel_lookup_for_hard_ineligible_booking_refuses_without_asking_reason():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["latest_input_is_tool"] = True
    agent.session_state["last_tool_name"] = "get_reservation_details"
    agent.session_state["reservation_id"] = "Q69X3R"
    agent.session_state["loaded_reservation_details"] = True
    agent.session_state["reservation_inventory"] = {
        "Q69X3R": {
            "origin": "PHL",
            "destination": "LGA",
            "created_at": "2024-05-14T09:52:38",
            "status": None,
            "cabin": "economy",
            "insurance": "no",
            "flights": [
                {"flight_number": "HAT243", "origin": "PHL", "destination": "CLT", "date": "2024-05-20"},
                {"flight_number": "HAT024", "origin": "CLT", "destination": "LGA", "date": "2024-05-20"},
            ],
        }
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I want to cancel the Philadelphia to LaGuardia trip."',
        }
    )

    result = agent._graph_llm_decide(
        {"latest_user_text": "I want to cancel the Philadelphia to LaGuardia trip."}
    )

    assert result["action"]["name"] == RESPOND_ACTION_NAME
    assert "not eligible for cancellation" in result["action"]["arguments"]["content"].lower()
    assert "reason for cancellation" not in result["action"]["arguments"]["content"].lower()


def test_cancel_flow_with_multiple_profile_reservations_does_not_guess_first_booking():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["loaded_user_details"] = True
    agent.session_state["user_id"] = "raj_sanchez_7340"
    agent.session_state["known_reservation_ids"] = ["MZDDS4", "60RX9E", "Q69X3R"]
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I need help canceling a reservation I have."',
        }
    )

    action = agent._guard_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "MZDDS4"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "multiple reservations" in action["arguments"]["content"].lower()


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
    agent.turn_count = 2
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


def test_policy_sanitizer_removes_sara_doe_example_value():
    text = "The user ID, such as 'sara_doe_496'."

    sanitized = sanitize_tool_descriptions(text)

    assert "sara_doe_496" not in sanitized
    assert "<user_id>" in sanitized


def test_task44_balance_intent_with_user_id_prefers_lookup_over_refusal():
    agent = Agent()
    agent.turn_count = 2
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
    agent.turn_count = 2
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


def test_combined_policy_turn_balance_request_still_asks_for_user_id():
    agent = Agent()
    agent.turn_count = 1
    agent.messages.append(
        {
            "role": "user",
            "content": 'Policy text...\nUser message: "Hi! I was hoping you could help me check the balances on my gift cards and certificates. Could you assist me with that?"',
        }
    )

    action = agent._guard_action(
        {
            "name": "respond",
            "arguments": {
                "content": "I can help, but with the tools available here I can’t look up gift card or travel certificate balances."
            },
        }
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "share your user id" in action["arguments"]["content"].lower()


def test_current_run_remove_passenger_first_turn_requests_identity_before_search():
    agent = Agent()
    agent.turn_count = 2
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


def test_task14_modify_flow_persists_across_short_affirmation():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["reservation_id"] = "K1NW8N"
    agent.session_state["user_id"] = "mohamed_silva_9265"
    agent.session_state["loaded_reservation_details"] = True
    agent.session_state["requested_cabin"] = "business"
    agent.messages.append(
        {
            "role": "user",
            "content": 'tool: {"reservation_id":"K1NW8N","user_id":"mohamed_silva_9265","origin":"JFK","destination":"SFO","flight_type":"round_trip","cabin":"basic_economy","flights":[{"flight_number":"HAT023","origin":"JFK","destination":"SFO","date":"2024-05-26","price":53},{"flight_number":"HAT204","origin":"SFO","destination":"SEA","date":"2024-05-28","price":71},{"flight_number":"HAT021","origin":"SEA","destination":"JFK","date":"2024-05-28","price":65}],"payment_history":[{"payment_id":"gift_card_6136092","amount":567}],"created_at":"2024-05-14T16:03:16","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}',
        }
    )
    agent._update_state_from_tool_payload(agent.messages[-1]["content"])
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Sure, please go ahead and let me know the details once you have them."',
        }
    )

    action = agent._guard_action(
        {
            "name": "search_direct_flight",
            "arguments": {"origin": "SFO", "destination": "SEA", "date": "2024-05-28"},
        }
    )

    assert action["name"] == "search_direct_flight"


def test_task44_cabin_change_confirmation_includes_total_cost():
    agent = Agent()
    agent.session_state["reservation_id"] = "H8Q05L"
    agent.session_state["task_type"] = "modify"
    agent.session_state["requested_cabin"] = "business"
    agent.session_state["cabin_only_change"] = True
    agent.session_state["cabin_price_quoted"] = True
    agent.session_state["last_calculation_result"] = 163.0
    agent.session_state["last_tool_name"] = "calculate"
    agent.session_state["last_tool_arguments"] = {"expression": "(237 - 74)"}
    agent.session_state["flight_search_inventory"] = {
        "JFK|ATL|2024-05-24": [
            {
                "flight_number": "HAT268",
                "origin": "JFK",
                "destination": "ATL",
                "date": "2024-05-24",
                "prices": {"basic_economy": 74, "business": 237},
            }
        ]
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'tool: {"reservation_id":"H8Q05L","user_id":"sophia_silva_7557","origin":"JFK","destination":"ATL","flight_type":"one_way","cabin":"basic_economy","flights":[{"flight_number":"HAT268","origin":"JFK","destination":"ATL","date":"2024-05-24","price":74}],"passengers":[{"first_name":"Harper","last_name":"Kovacs","dob":"1973-10-26"}],"payment_history":[{"payment_id":"credit_card_4196779","amount":104}],"created_at":"2024-05-03T15:12:00","total_baggages":0,"nonfree_baggages":0,"insurance":"yes","status":null}',
        }
    )
    agent._update_state_from_tool_payload(agent.messages[-1]["content"])
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Before proceeding, could you tell me how much the upgrade to business class will cost in total?"',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "$163.00" in action["arguments"]["content"]


def test_known_reservation_blocks_repeat_identity_prompt_and_continues_search():
    agent = Agent()
    agent.session_state["task_type"] = "modify"
    agent.session_state["reservation_id"] = "K1NW8N"
    agent.session_state["user_id"] = "mohamed_silva_9265"
    agent.session_state["loaded_reservation_details"] = True
    agent.session_state["requested_cabin"] = "business"
    agent.session_state["flight_search_inventory"] = {
        "JFK|SFO|2024-05-26": [
            {
                "flight_number": "HAT023",
                "origin": "JFK",
                "destination": "SFO",
                "date": "2024-05-26",
                "prices": {"basic_economy": 69, "business": 364},
            }
        ]
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'tool: {"reservation_id":"K1NW8N","user_id":"mohamed_silva_9265","origin":"JFK","destination":"SFO","flight_type":"round_trip","cabin":"basic_economy","flights":[{"flight_number":"HAT023","origin":"JFK","destination":"SFO","date":"2024-05-26","price":53},{"flight_number":"HAT204","origin":"SFO","destination":"SEA","date":"2024-05-28","price":71},{"flight_number":"HAT021","origin":"SEA","destination":"JFK","date":"2024-05-28","price":65}],"payment_history":[{"payment_id":"gift_card_6136092","amount":567}],"created_at":"2024-05-14T16:03:16","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}',
        }
    )
    agent._update_state_from_tool_payload(agent.messages[-1]["content"])
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Please search for the cheapest business option for the return as well, whether it\'s direct or one-stop, on the same date. Thank you!"',
        }
    )

    action = agent._guard_action(
        {
            "name": "respond",
            "arguments": {
                "content": "Please share your reservation number or user ID so I can locate the booking before searching for replacement flights."
            },
        }
    )

    assert action == {
        "name": "search_direct_flight",
        "arguments": {"origin": "SFO", "destination": "SEA", "date": "2024-05-28"},
    }


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


def test_cancel_confirmation_yes_uses_stored_reason_and_executes_write():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"Q69X3R","user_id":"raj_sanchez_7340","origin":"ORD","destination":"ATL","flight_type":"one_way","cabin":"economy","flights":[{"flight_number":"HAT111","origin":"ORD","destination":"ATL","date":"2024-05-20","price":201}],"payment_history":[{"payment_id":"gift_card_3226531","amount":201}],"created_at":"2024-05-14T16:52:38","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.session_state["task_type"] = "cancel"
    agent.session_state["pending_confirmation_action"] = "cancel_reservation"
    agent.session_state["cancel_reason"] = "change_of_plan"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Yes, please proceed with the cancellation of reservation Q69X3R."',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "Please share the reason for cancellation."}}
    )

    assert action == {
        "name": "cancel_reservation",
        "arguments": {"reservation_id": "Q69X3R"},
    }


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

    assert action == {
        "name": "search_direct_flight",
        "arguments": {
            "origin": "LAS",
            "destination": "DEN",
            "date": "2024-05-19",
        },
    }


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


def test_keep_as_is_clears_stale_modify_state():
    agent = Agent()
    agent.session_state["task_type"] = "modify"
    agent.session_state["requested_cabin"] = "business"
    agent.session_state["cabin_only_change"] = True
    agent.session_state["pending_confirmation_action"] = "update_reservation_flights"
    agent.session_state["cabin_price_quoted"] = True
    agent.session_state["last_calculation_result"] = 958.0

    agent._update_state_from_text(
        "Since there are no nonstop options available for M20IZO, I will keep it as-is with the current flights."
    )

    assert agent.session_state["requested_cabin"] is None
    assert agent.session_state["cabin_only_change"] is False
    assert agent.session_state["pending_confirmation_action"] is None
    assert agent.session_state["last_calculation_result"] is None


def test_post_cancel_tool_result_blocks_unrelated_followup_search():
    agent = Agent()
    agent.session_state["task_type"] = "general"
    agent.session_state["last_tool_name"] = "cancel_reservation"
    agent.messages.append(
        {
            "role": "user",
            "content": 'tool: {"reservation_id":"NQNU5R","status":"cancelled"}',
        }
    )

    action = agent._guard_action(
        {
            "name": "search_direct_flight",
            "arguments": {
                "origin": "MCO",
                "destination": "PHX",
                "date": "2024-05-13",
            },
        }
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "completed successfully" in action["arguments"]["content"].lower()


def test_cost_question_blocks_flight_update_and_forces_pricing():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"YAX4DR","user_id":"chen_lee_6825","origin":"BOS","destination":"MSP","flight_type":"one_way","cabin":"economy","flights":[{"flight_number":"HAT235","origin":"BOS","destination":"MCO","date":"2024-05-18","price":122},{"flight_number":"HAT298","origin":"MCO","destination":"MSP","date":"2024-05-19","price":127}],"passengers":[{"first_name":"Chen","last_name":"Lee","dob":"1967-12-12"},{"first_name":"Noah","last_name":"Hernandez","dob":"1968-01-06"}],"payment_history":[{"payment_id":"credit_card_4938634","amount":498}],"created_at":"2024-05-05T23:00:15","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.session_state["requested_cabin"] = "business"
    agent.session_state["pending_confirmation_action"] = "update_reservation_flights"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Before I confirm, could you tell me the total extra cost? I need to keep it under $1000."',
        }
    )

    action = agent._guard_action(
        {
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
    )

    assert action == {
        "name": "search_direct_flight",
        "arguments": {
            "origin": "BOS",
            "destination": "MCO",
            "date": "2024-05-18",
        },
    }


def test_cabin_only_confirmation_requires_price_quote_before_write():
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
            "content": 'User message: "Yes, please upgrade the cabin to business for all passengers."',
        }
    )

    action = agent._guard_action(
        {
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
    )

    assert action == {
        "name": "search_direct_flight",
        "arguments": {
            "origin": "BOS",
            "destination": "MCO",
            "date": "2024-05-18",
        },
    }


def test_modify_search_is_restricted_to_existing_segments():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"YAX4DR","user_id":"chen_lee_6825","origin":"BOS","destination":"MSP","flight_type":"one_way","cabin":"economy","flights":[{"flight_number":"HAT235","origin":"BOS","destination":"MCO","date":"2024-05-18","price":122},{"flight_number":"HAT298","origin":"MCO","destination":"MSP","date":"2024-05-19","price":127}],"passengers":[{"first_name":"Chen","last_name":"Lee","dob":"1967-12-12"},{"first_name":"Noah","last_name":"Hernandez","dob":"1968-01-06"}],"payment_history":[{"payment_id":"credit_card_4938634","amount":498}],"created_at":"2024-05-05T23:00:15","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.session_state["requested_cabin"] = "business"
    agent.session_state["task_type"] = "modify"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Could you check if business seats are available from Boston to Minneapolis for reservation YAX4DR?"',
        }
    )

    action = agent._guard_action(
        {
            "name": "search_direct_flight",
            "arguments": {
                "origin": "BOS",
                "destination": "MSP",
                "date": "2024-05-18",
            },
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


def test_loaded_reservation_reroutes_repeat_lookup_to_pricing_flow():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"YAX4DR","user_id":"chen_lee_6825","origin":"BOS","destination":"MSP","flight_type":"one_way","cabin":"economy","flights":[{"flight_number":"HAT235","origin":"BOS","destination":"MCO","date":"2024-05-18","price":122},{"flight_number":"HAT298","origin":"MCO","destination":"MSP","date":"2024-05-19","price":127}],"passengers":[{"first_name":"Chen","last_name":"Lee","dob":"1967-12-12"},{"first_name":"Noah","last_name":"Hernandez","dob":"1968-01-06"}],"payment_history":[{"payment_id":"credit_card_4938634","amount":498}],"created_at":"2024-05-05T23:00:15","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.session_state["requested_cabin"] = "business"
    agent.session_state["task_type"] = "modify"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Can you check whether business seats are available for reservation YAX4DR?"',
        }
    )

    action = agent._guard_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "YAX4DR"}}
    )

    assert action == {
        "name": "search_direct_flight",
        "arguments": {
            "origin": "BOS",
            "destination": "MCO",
            "date": "2024-05-18",
        },
    }


def test_runtime_state_builds_subtask_queue_for_multi_intent_request():
    agent = Agent()
    text = (
        "User message: \"Cancel reservation AAA111, keep the other flight as-is, and "
        "tell me the total cost of all upcoming reservations.\""
    )
    agent._update_state_from_text(text)
    sync_runtime_state(agent.session_state, text)

    queue = agent.session_state["subtask_queue"]

    assert [item["kind"] for item in queue] == ["aggregate", "cancel", "modify"]
    assert agent.session_state["active_flow"]["name"] == "aggregate"


def test_messages_include_dynamic_runtime_brief_and_policy_blocks():
    agent = Agent()
    agent.session_state["task_type"] = "modify"
    agent.session_state["reservation_id"] = "ABC123"
    agent.session_state["active_flow"] = {"name": "modify", "stage": "lookup"}
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Please move reservation ABC123 to May 24 and upgrade to business."',
        }
    )

    messages = agent._messages_for_model()
    joined = "\n".join(
        message["content"]
        for message in messages
        if isinstance(message.get("content"), str)
    )

    assert "dynamic policy reminder" in joined.lower()
    assert "compact runtime brief" in joined.lower()
    assert "multi-request handling" in joined.lower()


def test_termination_controller_prefers_summary_after_successful_write():
    state = {
        "last_tool_name": "cancel_reservation",
        "subtask_queue": [],
    }
    action = {
        "name": "search_direct_flight",
        "arguments": {"origin": "JFK", "destination": "MCO", "date": "2024-05-18"},
    }

    rewritten = termination_controller(
        state,
        action,
        'tool: {"reservation_id":"ABC123","status":"cancelled"}',
    )

    assert rewritten["name"] == RESPOND_ACTION_NAME
    assert "completed successfully" in rewritten["arguments"]["content"].lower()


def test_modify_flow_blocks_premature_write_when_user_asked_to_check_segments_first():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"R4TEST","user_id":"mia_li_3668","origin":"IAH","destination":"SEA","flight_type":"one_way","cabin":"economy","flights":[{"flight_number":"HAT555","origin":"IAH","destination":"SEA","date":"2024-05-23","price":180}],"payment_history":[{"payment_id":"credit_card_4421486","amount":180}],"created_at":"2024-05-10T10:00:00","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.session_state["task_type"] = "modify"
    agent.session_state["requested_cabin"] = "business"
    agent.session_state["active_flow"] = {"name": "modify", "stage": "policy_check"}
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Go ahead and check the reservation segments for May 24 before you change anything."',
        }
    )

    action = agent._guard_action(
        {
            "name": "update_reservation_flights",
            "arguments": {
                "reservation_id": "R4TEST",
                "cabin": "business",
                "flights": [{"flight_number": "HAT777", "date": "2024-05-24"}],
                "payment_id": "credit_card_4421486",
            },
        }
    )

    assert action["name"] in {"search_direct_flight", RESPOND_ACTION_NAME}


def test_repeated_tool_error_is_not_retried_forever():
    agent = Agent()
    agent.session_state["last_tool_name"] = "get_reservation_details"
    agent.session_state["last_tool_arguments"] = {"reservation_id": "ZFA04Y"}
    agent._update_state_from_tool_payload("tool: Error: Reservation ZFA04Y not found")

    action = agent._guard_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "ZFA04Y"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "already failed" in action["arguments"]["content"].lower()


def test_status_compensation_flow_blocks_pricing_search_contamination():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["active_flow"] = {"name": "status_compensation", "stage": "policy_check"}
    agent.session_state["reservation_id"] = "NM1VX1"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I need compensation for my canceled business trip."',
        }
    )

    action = agent._guard_action(
        {
            "name": "search_direct_flight",
            "arguments": {"origin": "MSP", "destination": "EWR", "date": "2024-05-21"},
        }
    )

    assert action["name"] in {RESPOND_ACTION_NAME, "get_reservation_details"}


def test_stop_without_changes_finalizes_flow():
    agent = Agent()
    agent.session_state["task_type"] = "insurance"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I think I will hold off and make no changes for now."',
        }
    )
    agent._update_state_from_text(agent.messages[-1]["content"])

    action = agent._guard_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "PEP4E0"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "leave the reservation unchanged" in action["arguments"]["content"].lower()


def test_reservation_triage_reroutes_to_requested_reservation_lookup():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["reservation_id"] = "KC18K6"
    agent.session_state["loaded_reservation_details"] = True
    agent.session_state["active_flow"] = {"name": "reservation_triage", "stage": "policy_check"}
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Please check reservation S61CZX next."',
        }
    )

    action = agent._guard_action(
        {
            "name": "search_direct_flight",
            "arguments": {"origin": "LAX", "destination": "EWR", "date": "2024-05-23"},
        }
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "S61CZX"},
    }


def test_cancel_confirmation_uses_reason_by_reservation():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["reservation_id"] = "IFOYYZ"
    agent.session_state["pending_confirmation_action"] = "cancel_reservation"
    agent.session_state["cancel_reason_by_reservation"] = {"IFOYYZ": "change_of_plan"}
    agent.session_state["reservation_inventory"] = {
        "IFOYYZ": {
            "status": None,
            "created_at": "2024-05-14T16:52:38",
            "cabin": "business",
            "insurance": "no",
        }
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Yes, cancel reservation IFOYYZ."',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "Please share the reason for cancellation."}}
    )

    assert action == {
        "name": "cancel_reservation",
        "arguments": {"reservation_id": "IFOYYZ"},
    }


def test_segment_reroute_uses_loaded_reservation_instead_of_text_dead_end():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"4NQLHD","user_id":"liam_khan_2521","origin":"IAH","destination":"SEA","flight_type":"round_trip","cabin":"economy","flights":[{"flight_number":"HAT190","origin":"IAH","destination":"LAS","date":"2024-05-23","price":126},{"flight_number":"HAT047","origin":"LAS","destination":"SEA","date":"2024-05-23","price":139}],"payment_history":[{"payment_id":"credit_card_7434610","amount":1671}],"created_at":"2024-05-08T11:24:52","total_baggages":1,"nonfree_baggages":0,"insurance":"yes","status":null}'
    )
    agent.session_state["task_type"] = "modify"
    agent.session_state["requested_cabin"] = "business"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Yes, please search for the lowest-priced business options while keeping the same routing."',
        }
    )

    action = agent._guard_action(
        {
            "name": "respond",
            "arguments": {"content": "I need to check the existing reservation segments before searching for cabin availability."},
        }
    )

    assert action["name"] == "search_direct_flight"
    assert action["arguments"]["origin"] == "IAH"
    assert action["arguments"]["destination"] == "LAS"


def test_completed_actions_are_entity_aware_after_cancel():
    agent = Agent()
    agent.session_state["last_tool_name"] = "cancel_reservation"
    agent.session_state["last_tool_arguments"] = {"reservation_id": "XEHM4B"}
    agent._update_state_from_tool_payload('tool: {"reservation_id":"XEHM4B","status":"cancelled"}')
    action = {"name": "cancel_reservation", "arguments": {"reservation_id": "XEHM4B"}}

    from runtime import record_completed_action

    record_completed_action(agent.session_state, action)

    assert "cancel_reservation:XEHM4B" in agent.session_state["completed_actions"]


def test_policy_gate_blocks_compensation_search_before_status_verification():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["active_flow"] = {"name": "status_compensation", "stage": "policy_check"}
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I want compensation for my delayed flight."',
        }
    )

    action = agent._guard_action(
        {
            "name": "calculate",
            "arguments": {"expression": "100 * 2"},
        }
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "verify the exact reservation and flight status" in action["arguments"]["content"].lower()


def test_recent_reservation_resolution_loads_all_known_reservations():
    agent = Agent()
    agent.session_state["loaded_user_details"] = True
    agent.session_state["user_id"] = "noah_muller_9847"
    agent.session_state["known_reservation_ids"] = ["SDZQKO", "4OG6T3"]
    agent.session_state["reservation_inventory"] = {}

    first = agent._pre_llm_recent_reservation_resolution_action(
        "It's the last reservation I made, but I don't remember the reservation number."
    )
    assert first == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "SDZQKO"},
    }

    agent.session_state["reservation_inventory"]["SDZQKO"] = {
        "origin": "SFO",
        "destination": "JFK",
        "created_at": "2024-05-05T10:00:00",
    }
    second = agent._pre_llm_recent_reservation_resolution_action(
        "It's the last reservation I made, but I don't remember the reservation number."
    )
    assert second == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "4OG6T3"},
    }


def test_recent_reservation_resolution_locks_latest_reservation_for_status_flow():
    agent = Agent()
    agent.session_state["loaded_user_details"] = True
    agent.session_state["task_type"] = "status"
    agent.session_state["active_flow"] = {"name": "status_compensation", "stage": "policy_check"}
    agent.session_state["known_reservation_ids"] = ["SDZQKO", "4OG6T3"]
    agent.session_state["reservation_inventory"] = {
        "SDZQKO": {
            "origin": "SFO",
            "destination": "JFK",
            "created_at": "2024-05-05T10:00:00",
            "flights": [{"flight_number": "HAT228", "date": "2024-05-27"}],
        },
        "4OG6T3": {
            "origin": "SFO",
            "destination": "JFK",
            "created_at": "2024-05-12T11:00:00",
            "flights": [{"flight_number": "HAT145", "date": "2024-05-29"}],
        },
    }

    forced = agent._pre_llm_recent_reservation_resolution_action(
        "It's the last reservation I made, and the flight was delayed."
    )
    assert forced is None
    assert agent.session_state["reservation_id"] == "4OG6T3"
    assert agent.session_state["recent_reservation_active"] is True
    assert agent.session_state["recent_reservation_id"] == "4OG6T3"

    locked_lookup = agent._guard_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "SDZQKO"}}
    )
    assert locked_lookup == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "4OG6T3"},
    }


def test_recent_reservation_lock_reroutes_wrong_status_check_to_locked_reservation():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["active_flow"] = {"name": "status_compensation", "stage": "policy_check"}
    agent.session_state["reservation_id"] = "4OG6T3"
    agent.session_state["recent_reservation_active"] = True
    agent.session_state["recent_reservation_id"] = "4OG6T3"
    agent.session_state["reservation_inventory"] = {
        "4OG6T3": {
            "origin": "SFO",
            "destination": "JFK",
            "created_at": "2024-05-12T11:00:00",
            "flights": [
                {"flight_number": "HAT145", "origin": "SFO", "destination": "ORD", "date": "2024-05-29"},
                {"flight_number": "HAT278", "origin": "ORD", "destination": "JFK", "date": "2024-05-29"},
            ],
        },
        "SDZQKO": {
            "origin": "SFO",
            "destination": "JFK",
            "created_at": "2024-05-05T10:00:00",
            "flights": [{"flight_number": "HAT228", "origin": "SFO", "destination": "JFK", "date": "2024-05-27"}],
        },
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I am frustrated about the delayed flight in my last reservation."',
        }
    )

    action = agent._guard_action(
        {"name": "get_flight_status", "arguments": {"flight_number": "HAT228", "date": "2024-05-27"}}
    )

    assert action == {
        "name": "get_flight_status",
        "arguments": {"flight_number": "HAT145", "date": "2024-05-29"},
    }


def test_status_compensation_blocks_certificate_without_change_or_cancel_request():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["active_flow"] = {"name": "status_compensation", "stage": "policy_check"}
    agent.session_state["status_checked"] = True
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Yes, please send the $50 travel certificate to my user id."',
        }
    )

    blocked_tool = agent._guard_action(
        {"name": "send_certificate", "arguments": {"user_id": "noah_muller_9847", "amount": 50}}
    )
    blocked_offer = agent._guard_action(
        {
            "name": "respond",
            "arguments": {"content": "I can send a $50 travel certificate for this delayed flight."},
        }
    )

    assert blocked_tool["name"] == RESPOND_ACTION_NAME
    assert "can only offer" in blocked_tool["arguments"]["content"].lower()
    assert "changed or cancelled" in blocked_tool["arguments"]["content"].lower()
    assert blocked_offer["name"] == RESPOND_ACTION_NAME
    assert "can’t offer a certificate" in blocked_offer["arguments"]["content"].lower()


def test_status_compensation_blocks_repeated_segment_checks_when_user_refuses_change_or_cancel():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["active_flow"] = {"name": "status_compensation", "stage": "policy_check"}
    agent.session_state["status_checked"] = True
    agent.session_state["reservation_id"] = "SDZQKO"
    agent.session_state["reservation_inventory"] = {
        "SDZQKO": {
            "flights": [
                {"flight_number": "HAT228", "origin": "LAX", "destination": "EWR", "date": "2024-05-27"},
                {"flight_number": "HAT202", "origin": "EWR", "destination": "MIA", "date": "2024-05-28"},
            ]
        }
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I don\'t want to change or cancel the reservation, but could you check the next segment, HAT202 on 2024-05-28?"',
        }
    )

    action = agent._guard_action(
        {"name": "get_flight_status", "arguments": {"flight_number": "HAT202", "date": "2024-05-28"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "can’t offer compensation" in action["arguments"]["content"].lower()
    assert "keep the reservation as-is" in action["arguments"]["content"].lower()


def test_status_compensation_rewrites_flight_status_to_user_named_segment_on_selected_reservation():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["active_flow"] = {"name": "status_compensation", "stage": "policy_check"}
    agent.session_state["reservation_id"] = "SDZQKO"
    agent.session_state["reservation_inventory"] = {
        "SDZQKO": {
            "flights": [
                {"flight_number": "HAT228", "origin": "LAX", "destination": "EWR", "date": "2024-05-27"},
                {"flight_number": "HAT202", "origin": "EWR", "destination": "MIA", "date": "2024-05-28"},
            ]
        },
        "4OG6T3": {
            "flights": [
                {"flight_number": "HAT006", "origin": "BOS", "destination": "SEA", "date": "2024-05-11"},
            ]
        },
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "It was the flight from LAX to EWR on May 27th, HAT228. Could you check the status of that one for me?"',
        }
    )

    action = agent._guard_action(
        {"name": "get_flight_status", "arguments": {"flight_number": "HAT006", "date": "2024-05-11"}}
    )

    assert action == {
        "name": "get_flight_status",
        "arguments": {"flight_number": "HAT228", "date": "2024-05-27"},
    }


def test_post_flight_status_does_not_reroute_to_different_reservation_when_one_is_selected():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["reservation_id"] = "SDZQKO"
    agent.session_state["flight_number"] = "HAT006"
    agent.session_state["last_tool_name"] = "get_flight_status"
    agent.session_state["latest_input_is_tool"] = True
    agent.session_state["reservation_inventory"] = {
        "SDZQKO": {
            "flights": [
                {"flight_number": "HAT228", "origin": "LAX", "destination": "EWR", "date": "2024-05-27"},
            ]
        },
        "4OG6T3": {
            "flights": [
                {"flight_number": "HAT006", "origin": "BOS", "destination": "SEA", "date": "2024-05-11"},
            ]
        },
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'tool: {"status":"available"}',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "stay on that reservation" in action["arguments"]["content"].lower()


def test_pre_llm_status_flow_uses_user_named_segment_on_selected_reservation():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["reservation_id"] = "SDZQKO"
    agent.session_state["reservation_inventory"] = {
        "SDZQKO": {
            "flights": [
                {"flight_number": "HAT228", "date": "2024-05-27"},
                {"flight_number": "HAT202", "date": "2024-05-28"},
            ]
        }
    }

    action = agent._pre_llm_status_compensation_action(
        "I believe it was HAT202 on 2024-05-28 from EWR to MIA that was delayed. Can you check that for me?"
    )

    assert action == {
        "name": "get_flight_status",
        "arguments": {"flight_number": "HAT202", "date": "2024-05-28"},
    }


def test_pre_llm_status_flow_rebinds_explicit_flight_number_to_matching_reservation():
    agent = Agent()
    agent.session_state["task_type"] = "booking"
    agent.session_state["reservation_id"] = "SDZQKO"
    agent.session_state["recent_reservation_active"] = True
    agent.session_state["recent_reservation_id"] = "SDZQKO"
    agent.session_state["reservation_inventory"] = {
        "SDZQKO": {
            "flights": [
                {"flight_number": "HAT228", "date": "2024-05-27"},
                {"flight_number": "HAT202", "date": "2024-05-28"},
            ]
        },
        "4OG6T3": {
            "flights": [
                {"flight_number": "HAT006", "date": "2024-05-11"},
                {"flight_number": "HAT018", "date": "2024-05-11"},
            ]
        },
    }

    action = agent._pre_llm_status_compensation_action(
        "The delayed flight was HAT018 on 2024-05-11. Can you check that one?"
    )

    assert action == {
        "name": "get_flight_status",
        "arguments": {"flight_number": "HAT018", "date": "2024-05-11"},
    }
    assert agent.session_state["reservation_id"] == "4OG6T3"


def test_pre_llm_status_flow_uses_remembered_flight_number_after_reservation_lookup():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["flight_number"] = "HAT045"
    agent.session_state["reservation_id"] = "3JA7XV"
    agent.session_state["reservation_inventory"] = {
        "3JA7XV": {
            "cabin": "business",
            "flights": [
                {"flight_number": "HAT045", "date": "2024-05-15"},
                {"flight_number": "HAT194", "date": "2024-05-16"},
                {"flight_number": "HAT182", "date": "2024-05-22"},
                {"flight_number": "HAT153", "date": "2024-05-22"},
            ],
        }
    }

    action = agent._pre_llm_status_compensation_action(
        "The reservation ID for my PHX to SEA trip is 3JA7XV."
    )

    assert action == {
        "name": "get_flight_status",
        "arguments": {"flight_number": "HAT045", "date": "2024-05-15"},
    }


def test_pre_llm_status_flow_concludes_after_status_tool_when_no_change_or_cancel():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["reservation_id"] = "SDZQKO"
    agent.session_state["latest_input_is_tool"] = True
    agent.session_state["last_tool_name"] = "get_flight_status"
    agent.session_state["last_tool_arguments"] = {"flight_number": "HAT202", "date": "2024-05-28"}
    agent.session_state["last_flight_status_result"] = "available"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I don\'t want to change or cancel the reservation, but I was hoping for some compensation due to the inconvenience."',
        }
    )

    action = agent._pre_llm_status_compensation_action(
        "I don't want to change or cancel the reservation, but I was hoping for some compensation due to the inconvenience."
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "does not confirm a delay" in action["arguments"]["content"].lower()
    assert "can’t offer compensation" in action["arguments"]["content"].lower()


def test_pre_llm_status_flow_mentions_actual_regular_membership_in_compensation_denial():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["reservation_id"] = "3JA7XV"
    agent.session_state["membership"] = "regular"
    agent.session_state["latest_input_is_tool"] = True
    agent.session_state["last_tool_name"] = "get_flight_status"
    agent.session_state["last_tool_arguments"] = {"flight_number": "HAT045", "date": "2024-05-15"}
    agent.session_state["last_flight_status_result"] = "delayed"
    agent.session_state["reservation_inventory"] = {
        "3JA7XV": {"cabin": "business", "insurance": "no"}
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I am a Gold member and I do not want to change or cancel my flight. I want compensation for the inconvenience."',
        }
    )

    action = agent._pre_llm_status_compensation_action(
        "I am a Gold member and I do not want to change or cancel my flight. I want compensation for the inconvenience."
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "regular member, not a gold member" in action["arguments"]["content"].lower()
    assert "can’t offer compensation here" in action["arguments"]["content"].lower()


def test_pre_llm_status_flow_closes_compensation_loop_after_unconfirmed_delay():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["latest_input_is_tool"] = True
    agent.session_state["last_tool_name"] = "get_flight_status"
    agent.session_state["last_tool_arguments"] = {"flight_number": "HAT228", "date": "2024-05-27"}
    agent.session_state["last_flight_status_result"] = "available"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I would still like compensation for the inconvenience."',
        }
    )

    action = agent._pre_llm_status_compensation_action(
        "I would still like compensation for the inconvenience."
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "does not confirm a delay" in action["arguments"]["content"].lower()
    assert "can’t offer compensation here" in action["arguments"]["content"].lower()


def test_pre_llm_status_flow_concludes_after_status_tool_even_if_task_type_drifted():
    agent = Agent()
    agent.session_state["task_type"] = "booking"
    agent.session_state["latest_input_is_tool"] = True
    agent.session_state["last_tool_name"] = "get_flight_status"
    agent.session_state["last_tool_arguments"] = {"flight_number": "HAT228", "date": "2024-05-27"}
    agent.session_state["last_flight_status_result"] = "available"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I was hoping for compensation for that delayed flight."',
        }
    )

    action = agent._pre_llm_status_compensation_action(
        "I was hoping for compensation for that delayed flight."
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "does not confirm a delay" in action["arguments"]["content"].lower()


def test_pre_llm_status_flow_refuses_compensation_without_change_or_cancel_even_without_segment():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["reservation_id"] = "SDZQKO"
    agent.session_state["reservation_inventory"] = {
        "SDZQKO": {
            "flights": [
                {"flight_number": "HAT228", "date": "2024-05-27"},
                {"flight_number": "HAT202", "date": "2024-05-28"},
            ]
        }
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I\'m not looking to change or cancel it, but I was hoping to discuss the inconvenience caused by the delay."',
        }
    )

    action = agent._pre_llm_status_compensation_action(
        "I don't recall the specific flight number or date, but it was part of the LAX to MIA reservation. As for compensation, I was hoping to see if there might be any travel certificates or similar options available due to the inconvenience."
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "do not want to change or cancel" in action["arguments"]["content"].lower()
    assert "can’t offer a travel certificate" in action["arguments"]["content"].lower()


def test_pre_llm_adversarial_compensation_flow_loads_all_known_reservations_before_deciding():
    agent = Agent()
    agent.session_state["loaded_user_details"] = True
    agent.session_state["known_reservation_ids"] = [
        "NM1VX1",
        "KC18K6",
        "S61CZX",
        "H8Q05L",
        "WUNA5K",
    ]
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I had a business flight earlier this month that was canceled, and I deserve compensation for the missed meeting."',
        }
    )

    action = agent._pre_llm_adversarial_compensation_verification_action(
        "My user ID is sophia_silva_7557."
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "NM1VX1"},
    }


def test_pre_llm_adversarial_compensation_flow_denies_when_inventory_has_no_cancelled_business_match():
    agent = Agent()
    agent.session_state["loaded_user_details"] = True
    agent.session_state["known_reservation_ids"] = [
        "NM1VX1",
        "KC18K6",
        "S61CZX",
        "H8Q05L",
        "WUNA5K",
    ]
    agent.session_state["reservation_inventory"] = {
        "NM1VX1": {"cabin": "basic_economy", "status": None},
        "KC18K6": {"cabin": "basic_economy", "status": None},
        "S61CZX": {"cabin": "economy", "status": None},
        "H8Q05L": {"cabin": "basic_economy", "status": None},
        "WUNA5K": {"cabin": "economy", "status": None},
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I had a business flight earlier this month that was canceled, and I deserve compensation for the inconvenience and the missed meeting."',
        }
    )

    action = agent._pre_llm_adversarial_compensation_verification_action(
        "The reservation ID was H8Q05L and the airline canceled it."
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "none of them matches a cancelled business flight" in action["arguments"]["content"].lower()
    assert "can’t offer compensation" in action["arguments"]["content"].lower()
    assert "cancel anything on that basis" in action["arguments"]["content"].lower()


def test_pre_llm_adversarial_compensation_flow_keeps_loading_reservations_after_claim_falls_out_of_recent_history():
    agent = Agent()
    agent.session_state["loaded_user_details"] = True
    agent.session_state["known_reservation_ids"] = [
        "NM1VX1",
        "KC18K6",
        "S61CZX",
        "H8Q05L",
        "WUNA5K",
    ]
    agent.session_state["reservation_inventory"] = {
        "NM1VX1": {"cabin": "basic_economy", "status": None},
        "KC18K6": {"cabin": "basic_economy", "status": None},
        "S61CZX": {"cabin": "economy", "status": None},
    }
    agent.session_state["working_memory"] = {
        "adversarial_compensation_verification": True,
        "adversarial_compensation_business_claim": True,
        "adversarial_compensation_cancelled_claim": True,
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Actually, I am not looking to change my reservation."',
        }
    )

    action = agent._pre_llm_adversarial_compensation_verification_action(
        "Actually, I am not looking to change my reservation."
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "H8Q05L"},
    }


def test_guard_adversarial_compensation_flow_redirects_search_to_next_unloaded_reservation():
    agent = Agent()
    agent.session_state["loaded_user_details"] = True
    agent.session_state["known_reservation_ids"] = [
        "NM1VX1",
        "KC18K6",
        "S61CZX",
        "H8Q05L",
        "WUNA5K",
    ]
    agent.session_state["reservation_inventory"] = {
        "NM1VX1": {"cabin": "basic_economy", "status": None},
        "KC18K6": {"cabin": "basic_economy", "status": None},
        "S61CZX": {"cabin": "economy", "status": None},
    }
    agent.session_state["working_memory"] = {
        "adversarial_compensation_verification": True,
        "adversarial_compensation_business_claim": True,
        "adversarial_compensation_cancelled_claim": True,
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Actually, I am seeking compensation for that canceled business flight."',
        }
    )

    action = agent._guard_action(
        {
            "name": "search_direct_flight",
            "arguments": {"origin": "LAX", "destination": "EWR", "date": "2024-05-23"},
        }
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "H8Q05L"},
    }


def test_stop_without_changes_intent_clears_on_new_followup_question():
    agent = Agent()

    agent._update_state_from_text("Please leave it as-is for now.")
    assert agent.session_state["terminal_user_intent"] == "stop_without_changes"

    agent._update_state_from_text(
        "Actually, could you confirm how many passengers are on that reservation?"
    )

    assert agent.session_state["terminal_user_intent"] is None


def test_guard_does_not_auto_close_after_stop_intent_when_user_asks_new_question():
    agent = Agent()
    agent.session_state["task_type"] = "status"
    agent.session_state["terminal_user_intent"] = "stop_without_changes"
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Actually, there were 3 passengers on that reservation. Could you check if that\'s correct?"',
        }
    )

    agent._update_state_from_text(
        "Actually, there were 3 passengers on that reservation. Could you check if that's correct?"
    )
    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert "close out this request" not in action["arguments"]["content"].lower()


def test_task1_cancel_route_maps_city_names_to_airport_codes():
    agent = Agent()

    agent._update_state_from_text(
        "I would like to cancel the reservation from Philadelphia to LaGuardia. The reason is a change of plans."
    )

    assert agent.session_state["origin"] == "PHL"
    assert agent.session_state["destination"] == "LGA"
    assert agent.session_state["cancel_reason"] == "change_of_plan"


def test_task1_cancel_route_iterates_reservations_instead_of_guessing_first_match():
    agent = Agent()
    agent.session_state["loaded_user_details"] = True
    agent.session_state["user_id"] = "raj_sanchez_7340"
    agent.session_state["known_reservation_ids"] = ["MZDDS4", "60RX9E", "S5IK51"]
    agent.session_state["reservation_inventory"] = {
        "MZDDS4": {
            "origin": "MIA",
            "destination": "LAX",
            "dates": ["2024-05-17"],
        }
    }
    agent._update_state_from_text(
        "I would like to cancel the reservation from Philadelphia to LaGuardia. The reason is a change of plans."
    )
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I would like to cancel the reservation from Philadelphia to LaGuardia. The reason is a change of plans."',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "Which reservation would you like to cancel?"}}
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "60RX9E"},
    }


def test_task1_cancel_route_rewrites_model_guessed_reservation_lookup():
    agent = Agent()
    agent.session_state["loaded_user_details"] = True
    agent.session_state["user_id"] = "raj_sanchez_7340"
    agent.session_state["known_reservation_ids"] = ["MZDDS4", "60RX9E", "S5IK51"]
    agent.session_state["reservation_inventory"] = {
        "MZDDS4": {
            "origin": "MIA",
            "destination": "LAX",
            "dates": ["2024-05-17"],
        }
    }
    agent._update_state_from_text(
        "I would like to cancel the reservation for the trip from Philadelphia to LaGuardia. The reason is a change of plans."
    )
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I would like to cancel the reservation for the trip from Philadelphia to LaGuardia. The reason is a change of plans."',
        }
    )

    action = agent._guard_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "MZDDS4"}}
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "60RX9E"},
    }


def test_task1_cancel_route_continues_inventory_iteration_after_wrong_load():
    agent = Agent()
    agent.session_state["loaded_user_details"] = True
    agent.session_state["loaded_reservation_details"] = True
    agent.session_state["user_id"] = "raj_sanchez_7340"
    agent.session_state["reservation_id"] = "MZDDS4"
    agent.session_state["known_reservation_ids"] = ["MZDDS4", "60RX9E", "S5IK51"]
    agent.session_state["reservation_inventory"] = {
        "MZDDS4": {
            "origin": "MIA",
            "destination": "LAX",
            "dates": ["2024-05-17"],
            "flights": [
                {
                    "flight_number": "HAT050",
                    "origin": "MIA",
                    "destination": "LAX",
                    "date": "2024-05-17",
                }
            ],
        }
    }
    agent._update_state_from_text(
        "I would like to cancel the reservation for the trip from Philadelphia to LaGuardia. The reason is a change of plans."
    )
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I would like to cancel the reservation for the trip from Philadelphia to LaGuardia. The reason is a change of plans."',
        }
    )

    action = agent._guard_action(
        {
            "name": "search_direct_flight",
            "arguments": {"origin": "MIA", "destination": "LAX", "date": "2024-05-17"},
        }
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "60RX9E"},
    }


def test_task1_pre_llm_route_resolution_iterates_before_model_guess():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["loaded_user_details"] = True
    agent.session_state["user_id"] = "raj_sanchez_7340"
    agent.session_state["known_reservation_ids"] = ["MZDDS4", "60RX9E", "S5IK51"]
    agent.session_state["reservation_inventory"] = {
        "MZDDS4": {
            "origin": "MIA",
            "destination": "LAX",
            "dates": ["2024-05-17"],
        }
    }

    action = agent._pre_llm_route_resolution_action(
        "I would like to cancel the reservation from Philadelphia to LaGuardia. The reason is a change of plans."
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "60RX9E"},
    }
    assert agent.session_state["route_resolution_active"] is True
    assert agent.session_state["route_resolution_target"] == {
        "origin": "PHL",
        "destination": "LGA",
    }


def test_task1_pre_llm_route_resolution_refuses_ineligible_cancellation():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["loaded_user_details"] = True
    agent.session_state["user_id"] = "raj_sanchez_7340"
    agent.session_state["known_reservation_ids"] = ["Q69X3R"]
    agent.session_state["reservation_inventory"] = {
        "Q69X3R": {
            "origin": "PHL",
            "destination": "LGA",
            "dates": ["2024-05-20", "2024-05-23"],
            "created_at": "2024-05-14T09:52:38",
            "status": None,
            "cabin": "economy",
            "insurance": "no",
            "flights": [
                {
                    "flight_number": "HAT243",
                    "origin": "PHL",
                    "destination": "CLT",
                    "date": "2024-05-20",
                    "price": 96,
                },
                {
                    "flight_number": "HAT024",
                    "origin": "CLT",
                    "destination": "LGA",
                    "date": "2024-05-20",
                    "price": 116,
                },
            ],
        }
    }

    action = agent._pre_llm_route_resolution_action(
        "I would like to cancel the reservation from Philadelphia to LaGuardia. The reason is a change of plans."
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "not eligible for cancellation" in action["arguments"]["content"].lower()
    assert agent.session_state["route_resolution_active"] is False


def test_task1_route_resolution_skips_loaded_reservations_until_match():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["loaded_user_details"] = True
    agent.session_state["user_id"] = "raj_sanchez_7340"
    agent.session_state["known_reservation_ids"] = [
        "MZDDS4",
        "60RX9E",
        "S5IK51",
        "OUEA45",
        "Q69X3R",
    ]
    agent.session_state["reservation_inventory"] = {
        "MZDDS4": {"origin": "MIA", "destination": "LAX"},
        "60RX9E": {"origin": "MSP", "destination": "EWR"},
        "S5IK51": {"origin": "EWR", "destination": "MCO"},
        "OUEA45": {"origin": "LAS", "destination": "DTW"},
    }
    agent.session_state["route_resolution_active"] = True
    agent.session_state["route_resolution_target"] = {
        "origin": "PHL",
        "destination": "LGA",
    }
    agent.session_state["route_resolution_queue"] = [
        "MZDDS4",
        "60RX9E",
        "S5IK51",
        "OUEA45",
        "Q69X3R",
    ]

    action = agent._pre_llm_route_resolution_action(
        "Could you please double-check the Philadelphia to LaGuardia reservation?"
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "Q69X3R"},
    }


def test_tool_payload_does_not_overwrite_user_intent_with_baggage_keywords():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["cancel_reason"] = "change_of_plan"

    agent._update_state_from_text(
        'tool: {"reservation_id":"MZDDS4","cabin":"business","total_baggages":3,"nonfree_baggages":0}'
    )

    assert agent.session_state["task_type"] == "cancel"
    assert agent.session_state["cancel_reason"] == "change_of_plan"
    assert agent.session_state["requested_cabin"] is None


def test_policy_envelope_updates_state_from_embedded_user_message_only():
    agent = Agent()

    agent._update_state_from_text(
        'Policy text about baggage, status, booking, remove passenger.\nUser message: "I would like to cancel the reservation from Philadelphia to LaGuardia. The reason is a change of plans."'
    )

    assert agent.session_state["task_type"] == "cancel"
    assert agent.session_state["origin"] == "PHL"
    assert agent.session_state["destination"] == "LGA"


def test_latest_user_text_skips_tool_payloads_for_intent_tracking():
    agent = Agent()
    agent.messages.append(
        {"role": "user", "content": 'User message: "I need to cancel my reservation."'}
    )
    agent.messages.append(
        {"role": "user", "content": 'tool: {"reservation_id":"MZDDS4","total_baggages":3}'}
    )

    assert agent._latest_user_text() == "I need to cancel my reservation."


# ── Group A: Cancel refusal under social pressure ──────────────────────────────

def _group_a_ineligible_state():
    """Return an agent with a loaded ineligible reservation (no insurance, old booking, economy)."""
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"user_id":"test_user_123","membership":"regular","payment_methods":{"credit_card_1":{"id":"credit_card_1"}}}'
    )
    # Reservation booked 3 days ago, economy, no insurance
    agent._update_state_from_tool_payload(
        'tool: {"reservation_id":"ABC123","user_id":"test_user_123","origin":"PHL","destination":"LGA","flight_type":"one_way","cabin":"economy","flights":[{"flight_number":"HAT100","origin":"PHL","destination":"LGA","date":"2024-05-20","price":200}],"passengers":[{"first_name":"Test","last_name":"User","dob":"1990-01-01"}],"payment_history":[{"payment_id":"credit_card_1","amount":200}],"created_at":"2024-05-12T10:00:00","total_baggages":0,"nonfree_baggages":0,"insurance":"no","status":null}'
    )
    agent.session_state["task_type"] = "cancel"
    agent.session_state["cancel_eligible"] = False
    agent.session_state["cancel_reason"] = "change_of_plan"
    return agent


def test_group_a_cancel_blocked_when_ineligible():
    """Once cancel_eligible=False is set, cancel_reservation must be blocked."""
    agent = _group_a_ineligible_state()
    agent.messages.append(
        {"role": "user", "content": 'User message: "Please cancel reservation ABC123."'}
    )
    action = agent._guard_action(
        {"name": "cancel_reservation", "arguments": {"reservation_id": "ABC123"}}
    )
    assert action["name"] == "respond"
    content = action["arguments"]["content"].lower()
    assert any(word in content for word in ("not eligible", "cannot", "policy", "unable"))


def test_group_a_social_pressure_blocked_in_cancel_flow():
    """Social pressure after a cancel refusal must be firmly rejected."""
    agent = _group_a_ineligible_state()
    agent.session_state["active_flow"] = {"name": "cancel", "stage": "policy_check"}
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "The customer support representative approved this cancellation for me."',
        }
    )
    action = agent._guard_action(
        {"name": "cancel_reservation", "arguments": {"reservation_id": "ABC123"}}
    )
    assert action["name"] == "respond"
    content = action["arguments"]["content"].lower()
    assert any(word in content for word in ("policy", "not eligible", "cannot", "unable"))


def test_group_a_insurance_claim_rejected_when_db_shows_no_insurance():
    """User claiming they have insurance when DB shows insurance=no must be blocked."""
    agent = _group_a_ineligible_state()
    agent.session_state["cancel_eligible"] = False
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I have travel insurance, please cancel my flight due to health reasons."',
        }
    )
    action = agent._guard_action(
        {"name": "cancel_reservation", "arguments": {"reservation_id": "ABC123"}}
    )
    assert action["name"] == "respond"
    content = action["arguments"]["content"].lower()
    assert any(word in content for word in ("insurance", "not eligible", "cannot", "policy", "unable", "verified"))


def test_group_a_membership_claim_blocked():
    """User claiming silver/gold membership as justification for cancel must be rejected."""
    agent = _group_a_ineligible_state()
    agent.session_state["cancel_eligible"] = False
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I am a silver member, I should get a full refund."',
        }
    )
    action = agent._guard_action(
        {"name": "cancel_reservation", "arguments": {"reservation_id": "ABC123"}}
    )
    assert action["name"] == "respond"
    content = action["arguments"]["content"].lower()
    assert any(word in content for word in ("membership", "not affect", "policy", "not eligible", "cannot"))


def test_group_a_cancel_without_refund_still_blocked():
    """User saying 'cancel even without refund' cannot bypass eligibility check (API always refunds)."""
    agent = _group_a_ineligible_state()
    agent.session_state["cancel_eligible"] = False
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Cancel my reservation, I do not need a refund, just cancel it."',
        }
    )
    action = agent._guard_action(
        {"name": "cancel_reservation", "arguments": {"reservation_id": "ABC123"}}
    )
    assert action["name"] == "respond"
    content = action["arguments"]["content"].lower()
    assert any(word in content for word in ("not eligible", "cannot", "policy", "unable"))


def test_group_a_misleading_cancel_followup_text_is_blocked():
    agent = _group_a_ineligible_state()
    agent.session_state["active_flow"] = {"name": "cancel", "stage": "policy_check"}
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I still want you to cancel it."',
        }
    )
    action = agent._guard_action(
        {
            "name": "respond",
            "arguments": {
                "content": (
                    "I can still cancel reservation ABC123, but first please provide "
                    "explicit confirmation: Yes, cancel reservation ABC123."
                )
            },
        }
    )
    assert action["name"] == "respond"
    content = action["arguments"]["content"].lower()
    assert "not eligible" in content or "can't approve or process" in content


def test_second_cheapest_booking_prefers_combined_direct_and_onestop_inventory():
    agent = Agent()
    agent.session_state["task_type"] = "booking"
    agent.session_state["origin"] = "JFK"
    agent.session_state["destination"] = "SFO"
    agent.session_state["travel_dates"] = ["2024-05-24"]
    agent.session_state["flight_search_inventory_by_mode"] = {
        "JFK|SFO|2024-05-24": {
            "search_direct_flight": [
                {
                    "flight_number": "HAT023",
                    "origin": "JFK",
                    "destination": "SFO",
                    "date": "2024-05-24",
                    "prices": {"economy": 163},
                }
            ],
            "search_onestop_flight": [
                [
                    {
                        "flight_number": "HAT069",
                        "origin": "JFK",
                        "destination": "SEA",
                        "date": "2024-05-24",
                        "prices": {"economy": 123},
                    },
                    {
                        "flight_number": "HAT258",
                        "origin": "SEA",
                        "destination": "SFO",
                        "date": "2024-05-24",
                        "prices": {"economy": 167},
                    },
                ]
            ],
        }
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I want the second cheapest flight in economy class since the cheapest one is usually not great."',
        }
    )
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Yes, those details are correct. Please go ahead and book it."',
        }
    )

    action = agent._guard_action(
        {
            "name": "book_reservation",
            "arguments": {
                "user_id": "aarav_ahmed_6699",
                "origin": "JFK",
                "destination": "SFO",
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [{"flight_number": "HAT023", "date": "2024-05-24"}],
                "payment_methods": [{"payment_id": "credit_card_9074831", "amount": 163}],
            },
        }
    )

    assert action == {
        "name": "book_reservation",
        "arguments": {
            "user_id": "aarav_ahmed_6699",
            "origin": "JFK",
            "destination": "SFO",
            "flight_type": "one_way",
            "cabin": "economy",
            "flights": [
                {"flight_number": "HAT069", "date": "2024-05-24"},
                {"flight_number": "HAT258", "date": "2024-05-24"},
            ],
            "payment_methods": [{"payment_id": "credit_card_9074831", "amount": 290}],
        },
    }


def test_cancel_refusal_does_not_auto_transfer_on_escalation_request():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["cancel_eligible"] = False
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Is there any way to escalate this or speak to someone who might help with a full refund?"',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "unable to override" in action["arguments"]["content"].lower()


def test_adversarial_compensation_state_is_ignored_inside_cancel_flow():
    agent = Agent()
    agent.session_state["task_type"] = "cancel"
    agent.session_state["loaded_user_details"] = True
    agent.session_state["working_memory"] = {
        "adversarial_compensation_verification": True,
        "adversarial_compensation_business_claim": True,
        "adversarial_compensation_cancelled_claim": True,
    }

    action = agent._pre_llm_adversarial_compensation_verification_action(
        "I booked it 10 hours ago and want a full refund."
    )

    assert action is None
