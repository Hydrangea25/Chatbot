function getBotResponse(input) {
  
    // Simple responses
    if (input == "hello") {
        return "Hello there heron!";
    } else if (input == "goodbye") {
        return "Talk to you later heron!";
    } else {
        return "Try asking something else!";
    }
}