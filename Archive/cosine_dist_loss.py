# for i_sent, input_id_1 in enumerate(input_ids):
            #     for i, input_id in enumerate(input_id_1):
            #         if input_id in attribute_list:
            #             for target_ids in target_ids_list:
            #                 if embedding_type == 'input':
            #                     target_embeds = all_input_embeds(
            #                         torch.LongTensor(target_ids))  # uncomment for input embed
            #                 elif embedding_type == 'output':
            #                     target_embeds = all_output_embeds[target_ids]
            #
            #                 debias_loss = torch.abs((1 - cos(hidden_states[i_sent, i], target_embeds[0])) -
            #                                         (1 - cos(hidden_states[i_sent, i], target_embeds[1])))
            #                 # print(debias_loss)
            #                 debias_loss_total = debias_loss_total + debias_loss